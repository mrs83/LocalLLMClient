import Foundation
import LocalLLMClientCore

/// A client for interacting with the Llama models.
///
/// This class provides methods for generating text streams from various inputs,
/// and handles the communication with the underlying Llama model.
public final class LlamaClient: LLMClient {
    private let context: Context
    private let multimodal: MultimodalContext?
    private let messageProcessor: MessageProcessor

    /// Guards the mutable tool state. `chatParams` is rebuilt by `setTools`,
    /// so reads and swaps must not race; retired pointers are freed only once
    /// no response stream can still reference them.
    private let stateLock = NSLock()
    nonisolated(unsafe) private var activeTools: [AnyLLMTool]
    /// Owned by this client; freed in `deinit` or when retired.
    nonisolated(unsafe) private var chatParams: UnsafeMutablePointer<llm_chat_params>?
    nonisolated(unsafe) private var retiredChatParams: [UnsafeMutablePointer<llm_chat_params>] = []
    nonisolated(unsafe) private var activeStreams = 0

    /// The tools currently configured for this client.
    var tools: [AnyLLMTool] {
        stateLock.lock()
        defer { stateLock.unlock() }
        return activeTools
    }

    var chatFormat: common_chat_format {
        stateLock.lock()
        defer { stateLock.unlock() }
        return get_chat_params_format(chatParams)
    }

    /// Initializes a new Llama client.
    ///
    /// - Parameters:
    ///   - url: The URL of the Llama model file.
    ///   - mmprojURL: The URL of the multimodal projector file (optional).
    ///   - parameter: The parameters for the Llama model.
    ///   - messageProcessor: The message processor to use for chat messages (optional).
    ///   - tools: An array of tools that can be used by the model for function calling.
    /// - Throws: An error if the client fails to initialize.
    public convenience init(
        url: URL,
        mmprojURL: URL?,
        parameter: Parameter,
        messageProcessor: MessageProcessor?,
        tools: [any LLMTool] = []
    ) throws {
        try self.init(
            url: url, mmprojURL: mmprojURL, parameter: parameter,
            messageProcessor: messageProcessor,
            erasedTools: tools.map { AnyLLMTool($0) })
    }

    /// Initializes a client with tools that are already type-erased.
    ///
    /// A host whose tools come from a runtime registry has no Swift type to
    /// derive a schema from, so it builds `AnyLLMTool` values directly. Erasing
    /// here would be the second erasure, and the first one is the one that
    /// cannot happen.
    public init(
        url: URL,
        mmprojURL: URL?,
        parameter: Parameter,
        messageProcessor: MessageProcessor?,
        erasedTools: [AnyLLMTool]
    ) throws {
        context = try Context(url: url, parameter: parameter)
        if let mmprojURL {
            multimodal = try MultimodalContext(url: mmprojURL, context: context, parameter: parameter)
        } else {
            multimodal = nil
        }
        self.messageProcessor = messageProcessor ?? MessageProcessorFactory.createAutoProcessor(chatTemplate: context.model.chatTemplate)
        self.activeTools = erasedTools
        self.chatParams = context.model.buildChatParams(tools: erasedTools)
    }

    deinit {
        if let chatParams {
            free_chat_params(chatParams)
        }
        for pointer in retiredChatParams {
            free_chat_params(pointer)
        }
    }

    /// Replaces the tool set used for chat-template rendering and tool-call
    /// parsing, without reloading the model or reallocating the context.
    ///
    /// The old chat parameters are retired and freed once no response stream
    /// can still reference them, so a long-running generation is never left
    /// with a dangling pointer.
    ///
    /// - Parameter tools: The tools available to the model for this request.
    public func setTools(_ tools: [AnyLLMTool]) {
        let newParams = context.model.buildChatParams(tools: tools)

        stateLock.lock()
        defer { stateLock.unlock() }
        activeTools = tools
        if let oldParams = chatParams {
            if activeStreams == 0 {
                free_chat_params(oldParams)
            } else {
                retiredChatParams.append(oldParams)
            }
        }
        chatParams = newParams
    }

    /// Drops the KV cache and prompt cache so the next request starts from an
    /// empty context.
    ///
    /// Callers that rebuild the whole transcript per request must use this:
    /// sliding-window / recurrent models cannot reliably trim the sequence to
    /// an arbitrary prefix, so the cached-prefix path lets the context grow
    /// request after request until `n_ctx` overflows.
    public func clearContext() {
        context.clear()
    }

    /// Marks a response stream as in flight and returns the chat parameters
    /// it must parse with.
    private func beginStream() -> UnsafeMutablePointer<llm_chat_params>? {
        stateLock.lock()
        defer { stateLock.unlock() }
        activeStreams += 1
        return chatParams
    }

    /// Releases the stream slot and frees chat parameters retired while the
    /// stream was running.
    private func endStream() {
        stateLock.lock()
        defer { stateLock.unlock() }
        activeStreams -= 1
        guard activeStreams == 0, !retiredChatParams.isEmpty else { return }
        for pointer in retiredChatParams {
            free_chat_params(pointer)
        }
        retiredChatParams.removeAll()
    }

    /// Generates a text stream from the given input.
    ///
    /// - Parameter input: The input to generate text from.
    /// - Returns: A generator that produces text as it's generated by the model.
    /// - Throws: An `LLMError.failedToDecode` error if the input cannot be decoded.
    public func textStream(from input: LLMInput) throws -> Generator {
        context.discardPendingTokens()
        let tools = self.tools

        do {
            switch input.value {
            case .plain(let text):
                context.clear()
                try context.decode(text: text)
            case .chatTemplate(let messages):
                try messageProcessor.process(
                    templateMessages: messages,
                    context: context,
                    multimodal: multimodal,
                    tools: tools
                )
            case .chat(let messages):
                try messageProcessor.process(
                    messages: messages,
                    context: context,
                    multimodal: multimodal,
                    tools: tools
                )
            }
        } catch {
            context.clear()
            throw LLMError.failedToDecode(reason: error.localizedDescription)
        }

        return Generator(context: context)
    }
    
    /// Streams responses from the input
    /// - Parameter input: The input to process
    /// - Returns: An asynchronous sequence that emits response content (text chunks, tool calls, etc.)
    /// - Throws: An error if generation fails
    public func responseStream(from input: LLMInput) async throws -> AsyncThrowingStream<StreamingChunk, any Error> {
        // Snapshot the chat parameters for this stream; `endStream` (below)
        // releases the slot so `setTools` can free retired pointers safely.
        // The pointer is only read inside the task and never escapes it.
        nonisolated(unsafe) let chatParams = beginStream()

        // Create the stream first (this can throw)
        let textStreamGenerator: Generator
        do {
            textStreamGenerator = try textStream(from: input)
        } catch {
            endStream()
            throw error
        }

        return AsyncThrowingStream { continuation in
            let processor = StreamingToolCallProcessor(
                startTag: getToolCallStartTag(),
                endTag: getToolCallEndTag()
            )

            let task = Task {
                defer { endStream() }
                do {
                    var fullText = ""

                    for try await chunk in textStreamGenerator {
                        fullText += chunk

                        // Process the chunk through the tool call processor
                        if let processedText = processor.processChunk(chunk) {
                            continuation.yield(.text(processedText))
                        }
                    }

                    let parserToolCalls = LlamaToolCallParser.parseToolCalls(
                        from: fullText,
                        chatParams: chatParams
                    ) ?? []
                    var toolCalls = processor.toolCalls + parserToolCalls
                    toolCalls = toolCalls.reduce(into: []) { result, toolCall in
                        if !result.contains(where: { $0.name == toolCall.name }) {
                            result.append(toolCall)
                        }
                    }

                    for toolCall in toolCalls {
                        continuation.yield(.toolCall(toolCall))
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            // A consumer that stops iterating (barge-in, new prompt, view
            // dismissed) must stop generation instead of decoding to EOS.
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
    
    /// Get the tool call start tag based on chat format
    private func getToolCallStartTag() -> String {
        switch chatFormat {
        case COMMON_CHAT_FORMAT_PEG_GEMMA4:
            return "<|tool_call>"
        default:
            return "<tool_call>"
        }
    }

    /// Get the tool call end tag based on chat format
    private func getToolCallEndTag() -> String {
        switch chatFormat {
        case COMMON_CHAT_FORMAT_PEG_GEMMA4:
            return "<tool_call|>"
        default:
            return "</tool_call>"
        }
    }
    
    /// Pauses any ongoing text generation
    public func pauseGeneration() async {
        await context.pauseHandler.pause()
    }
    
    /// Resumes previously paused text generation
    public func resumeGeneration() async {
        await context.pauseHandler.resume()
    }
    
    /// Whether the generation is currently paused
    public var isGenerationPaused: Bool {
        get async {
            await context.pauseHandler.isPaused
        }
    }
}

public extension LocalLLMClient {
    /// Creates a new Llama client.
    ///
    /// This is a factory method for creating `LlamaClient` instances.
    ///
    /// - Parameters:
    ///   - url: The URL of the Llama model file.
    ///   - mmprojURL: The URL of the multimodal projector file (optional).
    ///   - parameter: The parameters for the Llama model. Defaults to `.default`.
    ///   - messageProcessor: The message processor to use for chat messages (optional).
    ///   - tools: An array of tools that can be used by the model for function calling.
    ///   - verbose: A Boolean value indicating whether to enable verbose logging. Defaults to `false`.
    /// - Returns: A new `LlamaClient` instance.
    /// - Throws: An error if the client fails to initialize.
    static func llama(
        url: URL,
        mmprojURL: URL? = nil,
        parameter: LlamaClient.Parameter = .default,
        messageProcessor: MessageProcessor? = nil,
        tools: [any LLMTool] = []
    ) async throws -> LlamaClient {
        setLlamaVerbose(parameter.options.verbose)
        return try LlamaClient(
            url: url,
            mmprojURL: mmprojURL,
            parameter: parameter,
            messageProcessor: messageProcessor,
            tools: tools
        )
    }

    /// Creates a new Llama client from tools that are already type-erased.
    static func llama(
        url: URL,
        mmprojURL: URL? = nil,
        parameter: LlamaClient.Parameter = .default,
        messageProcessor: MessageProcessor? = nil,
        erasedTools: [AnyLLMTool]
    ) async throws -> LlamaClient {
        setLlamaVerbose(parameter.options.verbose)
        return try LlamaClient(
            url: url,
            mmprojURL: mmprojURL,
            parameter: parameter,
            messageProcessor: messageProcessor,
            erasedTools: erasedTools
        )
    }
}

#if DEBUG
extension LlamaClient {
    var _context: Context {
        context
    }

    var _multimodal: MultimodalContext? {
        multimodal
    }
}
#endif
