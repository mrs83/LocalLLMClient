import Foundation
import LocalLLMClientCore
import Testing

@testable import LocalLLMClientLlama

@Suite
struct ToolInstructionProcessorTests {
    private let processor = StandardToolInstructionProcessor()

    private var weatherTool: AnyLLMTool {
        AnyLLMTool(
            name: "get_weather",
            description: "Get the weather",
            argumentsSchema: ["type": "object", "properties": [String: any Sendable]()]
        ) { _ in
            ToolOutput(["result": "sunny"])
        }
    }

    @Test
    func nativeToolTemplatesAreDetected() {
        #expect(processor.hasNativeToolSupport(in: "{%- if tools %}{{ tools }}{%- endif %}"))
        #expect(processor.hasNativeToolSupport(in: "{{ tools }}"))
        #expect(processor.hasNativeToolSupport(in: "{{- tools | tojson }}"))
        #expect(processor.hasNativeToolSupport(in: "{% for tool in tools %}{{ tool }}{% endfor %}"))
    }

    @Test
    func proseMentionsAreNotNativeSupport() {
        #expect(!processor.hasNativeToolSupport(in: "You have access to tools."))
        #expect(!processor.hasNativeToolSupport(in: "{# tools are not supported #}"))
        #expect(!processor.hasNativeToolSupport(in: "{{ tools_list }}"))
    }

    @Test
    func nativeTemplatesKeepMessagesUntouched() throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "S"],
            ["role": "user", "content": "U"],
        ]

        let processed = try processor.processMessages(
            messages,
            tools: [weatherTool],
            templateHasNativeSupport: true
        )

        #expect(processed.count == messages.count)
        #expect(processed[0]["content"] as? String == "S")
        #expect(processed[0]["tools"] == nil)
    }

    @Test
    func fallbackTemplatesGetInstructions() throws {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "S"],
            ["role": "user", "content": "U"],
        ]

        let processed = try processor.processMessages(
            messages,
            tools: [weatherTool],
            templateHasNativeSupport: false
        )

        let system = processed[0]["content"] as? String ?? ""
        #expect(system.hasPrefix("S"))
        #expect(system.contains("<tool_call>"))
        #expect(system.contains("get_weather"))
    }
}
