#if BUILD_DOCC
@preconcurrency @_implementationOnly import llama
#elseif canImport(llama)
@preconcurrency private import llama
#else
@preconcurrency import LocalLLMClientLlamaC
#endif
import LocalLLMClientCore

public extension Context {
    @discardableResult
    func decode() throws(LLMError) -> Int32 {
        let numberOfTokens = batch.n_tokens
        guard batch.n_tokens > 0 else {
            return 0 // no data to decode
        }

        if parameter.context < position + batch.n_tokens {
            throw LLMError.failedToDecode(reason: "context size exceeded[\(parameter.context) < \(position + batch.n_tokens)]")
        }

        batch.logits[Int(batch.n_tokens) - 1] = 1

        guard llama_decode(context, batch) == 0 else {
            throw .failedToDecode(reason: "batch decode failed")
        }

        batch.clear()

        return numberOfTokens
    }

    func decode(text: String) throws(LLMError) {
        let tokens = [llama_token](text, addBos: false, special: true, vocab: vocab)
        lastPromptTokens = tokens
        try decode(tokens: tokens, from: 0)
    }

    /// Decodes a rendered chat prompt, keeping the KV cells it shares with the
    /// previous prompt and dropping only the tail.
    ///
    /// Hosts that rebuild the whole transcript per request (system prompt +
    /// tools + history + new turn) share a long prefix between requests. The
    /// chunk cache cannot see it — it compares whole chunks, and the previous
    /// generation sits on top of the sequence — so appending instead grows the
    /// context until it overflows. Comparing token IDs keeps the shared prefix
    /// evaluated and re-decodes only what changed.
    func decode(prompt text: String) throws(LLMError) {
        let tokens = [llama_token](text, addBos: false, special: true, vocab: vocab)
        let shared = Self.commonPrefixLength(lastPromptTokens, tokens)

        // Generated tokens sit after the last prompt, so everything past the
        // shared prefix goes even when the new prompt is a strict extension.
        if shared < position, let kv = llama_get_memory(context) {
            _ = llama_memory_seq_rm(kv, 0, llama_pos(shared), -1)
        }

        lastPromptTokens = tokens
        try decode(tokens: tokens, from: shared)
    }

    private func decode(tokens: [llama_token], from start: Int) throws(LLMError) {
        guard start < tokens.count else { return }
        let startPosition = position

        // Split tokens into chunks based on parameter.batch to avoid buffer overflow
        let chunkSize = parameter.batch
        for chunkStart in stride(from: start, to: tokens.count, by: chunkSize) {
            let chunk = tokens[chunkStart..<min(chunkStart + chunkSize, tokens.count)]

            let chunkOffset = startPosition + llama_pos(chunkStart - start)
            for (index, token) in chunk.enumerated() {
                batch.add(id: token, pos: chunkOffset + llama_pos(index), seq_ids: [0], logits: false)
            }
            try decode()
        }
    }

    private static func commonPrefixLength(_ lhs: [llama_token], _ rhs: [llama_token]) -> Int {
        let limit = min(lhs.count, rhs.count)
        var index = 0
        while index < limit, lhs[index] == rhs[index] {
            index += 1
        }
        return index
    }
}
