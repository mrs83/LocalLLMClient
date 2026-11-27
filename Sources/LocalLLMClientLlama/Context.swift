#if BUILD_DOCC
@preconcurrency @_implementationOnly import llama
#elseif canImport(llama)
@preconcurrency private import llama
#else
@preconcurrency import LocalLLMClientLlamaC
#endif
import Foundation
import LocalLLMClientCore

public final class Context: @unchecked Sendable {
    let parameter: LlamaClient.Parameter
    package let context: OpaquePointer
    package var batch: llama_batch
    var sampling: Sampler
    let grammer: Sampler?
    let cursorPointer: UnsafeMutableBufferPointer<llama_token_data>
    let model: Model
    let extraEOSTokens: Set<String>
    private var promptCaches: [(chunk: MessageChunk, lastPosition: llama_pos)] = []
    /// Tokens of the last prompt decoded into the sequence. Generated tokens
    /// occupy the positions after them, so the next prompt can keep the cells
    /// it shares with this one and drop only the tail.
    package var lastPromptTokens: [llama_token] = []
    let pauseHandler: PauseHandler
    /// Set when the host abandons a generation. `llama_decode` polls it through
    /// the context's abort callback, so a prompt being ingested aborts too
    /// instead of running to the end of a decode nobody reads.
    private let generationAborted: AbortFlag

    package var vocab: OpaquePointer {
        model.vocab
    }

    package var numberOfBatch: Int32 {
        Int32(llama_n_batch(context))
    }

    package var position: Int32 {
        guard let kv = llama_get_memory(context) else {
            return -1
        }

        return llama_memory_seq_pos_max(kv, 0) + 1
    }

    /// Keeps a couple of cores free on larger machines without starving small ones.
    private static var defaultNumberOfThreads: Int {
        let cores = ProcessInfo.processInfo.activeProcessorCount
        return cores > 4 ? min(8, cores - 2) : max(1, cores)
    }

    public init(url: URL, parameter: LlamaClient.Parameter = .default) throws(LLMError) {
        initializeLlama()

        let model = try Model(url: url)

        // A context of zero means "use the window the model was trained with".
        // The resolved value is stored so generation stops at the real limit
        // instead of at zero.
        var parameter = parameter
        if parameter.context <= 0 {
            parameter.context = model.trainedContextLength
        }

        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = UInt32(parameter.context)
        ctx_params.n_threads = Int32(parameter.numberOfThreads ?? Self.defaultNumberOfThreads)
        ctx_params.n_threads_batch = ctx_params.n_threads
        // A cancelled generation must stop inside `llama_decode`, not at the
        // next token: dropping the consumer of a long prompt used to leave the
        // whole prefill running to its end.
        let abortFlag = AbortFlag()
        ctx_params.abort_callback = { data in
            guard let data else { return false }
            return Unmanaged<AbortFlag>.fromOpaque(data).takeUnretainedValue().isSet
        }
        ctx_params.abort_callback_data = Unmanaged.passUnretained(abortFlag).toOpaque()

        self.parameter = parameter
        self.generationAborted = abortFlag
        self.pauseHandler = PauseHandler(disableAutoPause: parameter.options.disableAutoPause)
        self.model = model
        self.context = try model.makeAndAllocateContext(with: ctx_params)
        batch = llama_batch_init(Int32(parameter.batch), 0, 1)
        extraEOSTokens = parameter.options.extraEOSTokens

        // https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp
        sampling = llama_sampler_chain_init(llama_sampler_chain_default_params())

        if let format = parameter.options.responseFormat {
            switch format {
            case .json:
                do {
                    let template = try String(contentsOf: Bundle.module.url(forResource: "json", withExtension: "gbnf")!, encoding: .utf8)
                    grammer = llama_sampler_init_grammar(model.vocab, template, "root")
                } catch {
                    throw .failedToLoad(reason: "Failed to load grammar template")
                }
            case let .grammar(grammar, root):
                grammer = llama_sampler_init_grammar(model.vocab, grammar, root)
            }
            llama_sampler_chain_add(sampling, grammer)
        } else {
            grammer = nil
        }

        let minKeep = 0
        let penaltyFreq: Float = 0
        let penaltyPresent: Float = 0
        llama_sampler_chain_add(sampling, llama_sampler_init_temp_ext(parameter.temperature, 0, 1.0))
        llama_sampler_chain_add(sampling, llama_sampler_init_top_k(Int32(parameter.topK)))
        llama_sampler_chain_add(sampling, llama_sampler_init_top_p(parameter.topP, minKeep))
        llama_sampler_chain_add(sampling, llama_sampler_init_min_p(1 - parameter.topP, 1))
        llama_sampler_chain_add(sampling, llama_sampler_init_typical(parameter.typicalP, minKeep))
        llama_sampler_chain_add(sampling, llama_sampler_init_penalties(Int32(parameter.penaltyLastN), parameter.penaltyRepeat, penaltyFreq, penaltyPresent))
        llama_sampler_chain_add(sampling, llama_sampler_init_dist(parameter.seed.map(UInt32.init) ?? LLAMA_DEFAULT_SEED))

        cursorPointer = .allocate(capacity: Int(llama_vocab_n_tokens(model.vocab)))
    }

    deinit {
        cursorPointer.deallocate()
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_free(context)
    }

    /// Asks `llama_decode` to abandon whatever it is computing, prefill
    /// included, at its next check.
    public func cancelGeneration() {
        generationAborted.set()
    }

    /// Clears the abort flag so the next prompt is not cancelled by a stale
    /// request. Called as each prompt starts decoding.
    public func beginGeneration() {
        generationAborted.clear()
    }

    public func clear() {
        discardPendingTokens()
        lastPromptTokens.removeAll()

        guard let kv = llama_get_memory(context) else {
            return
        }

        llama_memory_clear(kv, true)
    }

    /// Drops tokens a generation left pending when it ended before decoding them.
    /// Appending the next prompt on top of them would write past the batch allocation.
    func discardPendingTokens() {
        batch.clear()
    }

    func addCache(for chunk: MessageChunk, position: llama_pos) {
        let endIndex = promptCaches.endIndex - 1
        switch (chunk, promptCaches.last?.chunk) {
        case let (.text(chunkText), .text(cacheText)):
            promptCaches[endIndex] = (chunk: .text(cacheText + chunkText), lastPosition: position)
        case let (.image(chunkImages), .image(cacheImages)):
            promptCaches[endIndex] = (chunk: .image(cacheImages + chunkImages), lastPosition: position)
        case let (.video(chunkVideos), .video(cacheVideos)):
            promptCaches[endIndex] = (chunk: .video(cacheVideos + chunkVideos), lastPosition: position)
        default:
            promptCaches.append((chunk: chunk, lastPosition: position))
        }
    }

    func removeCachedChunks(_ chunks: inout [MessageChunk]) {
        guard let (lastCacheIndex, newChunk) = lastCacheIndex(of: chunks) else {
            return
        }
        chunks = Array(chunks[(lastCacheIndex + 1)...])
        if let newChunk {
            chunks.append(newChunk)
        }
        if promptCaches[lastCacheIndex].lastPosition < position,
           let kv = llama_get_memory(context) {
            assert(llama_memory_seq_rm(kv, 0, promptCaches[lastCacheIndex].lastPosition, position))
        }
        if promptCaches.count > lastCacheIndex {
            promptCaches.removeSubrange((lastCacheIndex + 1)...)
        }
    }

    func lastCacheIndex(of chunks: [MessageChunk]) -> (index: Int, remaining: MessageChunk?)? {
        for (index, (chunk, cache)) in zip(chunks, promptCaches).enumerated() {
            switch (chunk, cache.chunk) {
            case let (.text(chunkText), .text(cacheText)) where chunkText.hasPrefix(cacheText):
                if chunkText == cacheText {
                    return (index, nil)
                } else {
                    return (index, .text(String(chunkText.dropFirst(cacheText.count))))
                }
            case let (.image(chunkImages), .image(cacheImages)) where chunkImages == cacheImages:
                return (index, nil)
            case let (.video(chunkVideos), .video(cacheVideos)) where chunkVideos == cacheVideos:
                return (index, nil)
            default:
                break
            }
        }
        return nil
    }
}


/// A thread-safe abort flag shared with llama.cpp's abort callback.
///
/// The callback is a C function pointer, so the flag has to be reachable
/// through `abort_callback_data` rather than captured.
package final class AbortFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flagged = false

    var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return flagged
    }

    func set() {
        lock.lock()
        flagged = true
        lock.unlock()
    }

    func clear() {
        lock.lock()
        flagged = false
        lock.unlock()
    }
}
