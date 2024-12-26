using AutoGen.Core;
using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Mistral;
using Microsoft.ML.Tokenizers;
using System.Text;
using static TorchSharp.torch;
using TorchSharp;
using Microsoft.ML.GenAI.Mistral.Module;
using System.Text.Json;
using TorchSharp.PyBridge;

namespace Local.NET.Mistral;

internal class Mistral(string weightFolder, string configName = "config.json", bool streamChat = true, bool useAgent = true)
{
    private static MistralFunctionCall instance = new ();

    internal async Task ChatAsync()
    {
        Console.WriteLine("Loading Mistral...");

        var (device, defaultType) = Util.InitializeTorch();
        var pipeline = Util.LoadMistralModel(device, weightFolder, configName);
        var agent = new MistralCausalLMAgent(pipeline, "assistant");

        if (useAgent)
        {
            await UseAgentAsync(agent, streamChat);
        }
        else
        {
            UsePipeline(pipeline);
        }
    }

    private async Task UseAgentAsync(MistralCausalLMAgent agent, bool stream)
    {
        Console.WriteLine("You are now chating with Mistral. To exit type '/exit'.");

        var messages = new List<TextMessage>();

        while (true)
        {
            Console.Write("User> ");

            var userQuery = Console.ReadLine();

            if (string.IsNullOrEmpty(userQuery) || userQuery.Contains("/exit"))
            {
                break;
            }

            var message = new TextMessage(new TextMessageUpdate(Role.User, userQuery));
            var mistralResponse = new StringBuilder();

            if (stream)
            {
                messages.Add(message);
                Console.Write("Mistral> ");

                await foreach (var resp in agent.GenerateStreamingReplyAsync(messages))
                {
                    mistralResponse.Append(resp.GetContent());
                    Console.Write(resp.GetContent());
                }

                Console.WriteLine();
            }
            else
            {
                var agentResponse = await agent.SendAsync(message, messages);

                Console.Write("Mistral> ");
                Console.WriteLine(agentResponse.GetContent());

                mistralResponse.Append(agentResponse.GetContent());
            }

            messages.Add(new TextMessage(new TextMessageUpdate(Role.Assistant, mistralResponse.ToString())));
        }

        Console.WriteLine("Bye!");
    }

    private void UsePipeline(CausalLMPipeline<LlamaTokenizer, MistralForCausalLM> pipeline)
    {
        Console.WriteLine("You are now chating with Mistral. To exit type '/exit'.");

        var templateBuilder = new Mistral_7B_0_3ChatTemplateBuilder();
        var messages = new List<TextMessage>
        {
            new TextMessage(Role.System, "You are a helpful assistant.")
        };

        while (true)
        {
            Console.Write("User> ");

            var userQuery = Console.ReadLine();

            if (string.IsNullOrEmpty(userQuery) || userQuery.Contains("/exit"))
            {
                break;
            }

            var message = new TextMessage(new TextMessageUpdate(Role.User, userQuery));
            var mistralResponse = new StringBuilder();

            messages.Add(message);

            Console.Write("Mistral> ");

            var maxLen = 1024;
            var temperature = 0.7f;
            var stopTokenSequence = new string[] { "</s>" };
            var template = templateBuilder.BuildPrompt(messages);

            foreach (string mistralMessage in pipeline.GenerateStreaming(prompt: template, maxLen: maxLen, temperature: temperature, stopSequences: stopTokenSequence))
            {
                mistralResponse.Append(mistralMessage);
                Console.Write(mistralMessage);
            }

            Console.WriteLine();

            messages.Add(new TextMessage(new TextMessageUpdate(Role.Assistant, mistralResponse.ToString())));
        }

        Console.WriteLine("Bye!");
    }

    public async Task WeatherChatAsync()
    {
        Console.WriteLine("Loading Mistral...");

        var (device, defaultType) = Util.InitializeTorch();
        var originalWeightFolder = Path.Combine(weightFolder);
        var pipeline = Util.LoadMistralModel(device, weightFolder, configName);
        var weatherChatMiddleware = new FunctionCallMiddleware(
            functions: [instance.GetWeatherFunctionContract],
            functionMap: new Dictionary<string, Func<string, Task<string>>>
            {
                { instance.GetWeatherFunctionContract.Name!, instance.GetWeatherWrapper }
            });
        var agent = new MistralCausalLMAgent(pipeline, "assistant")
            .RegisterStreamingMiddleware(weatherChatMiddleware); //.RegisterPrintMessage();
        var task = "what is the weather in Seattle";
        var userMessage = new TextMessage(Role.User, task);
        var reply = await agent.GenerateReplyAsync(messages: [userMessage],
            new GenerateReplyOptions
            {
                Temperature = 0f,
            });

        // generate further reply using tool call result;
        var message = await agent.SendAsync(chatHistory: [userMessage, reply]);

        Console.WriteLine($"Assistant: {message.GetContent()}");
    }

    public static void Embedding(string weightFolder = @"C:\Users\maxim\source\repos\models\BAAI\bge-en-icl")
    {
        var (device, defaultType) = Util.InitializeTorch();
        var configName = "config.json";
        var originalWeightFolder = Path.Combine(weightFolder);

        Console.WriteLine("Loading Mistral...");

        var tokenizer = MistralTokenizerHelper.FromPretrained(originalWeightFolder, modelName: "tokenizer.model");
        var mistralConfig = JsonSerializer.Deserialize<MistralConfig>(File.ReadAllText(Path.Combine(weightFolder, configName))) ?? throw new ArgumentNullException(nameof(configName));
        var model = new MistralModel(mistralConfig);

        model.load_checkpoint(weightFolder, "model.safetensors.index.json", strict: true, useTqdm: false);
        model.to(device);

        var pipeline = new CausalLMPipeline<LlamaTokenizer, MistralModel>(tokenizer, model, device);

        //var query = """
        //    <instruct>Given a web search query, retrieve relevant passages that answer the query.
        //    <query>what is a virtual interface
        //    <response>A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes.

        //    <instruct>Given a web search query, retrieve relevant passages that answer the query.
        //    <query>causes of back pain in female for a week
        //    <response>Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management.

        //    <instruct>Given a web search query, retrieve relevant passages that answer the query.
        //    <query>how much protein should a female eat
        //    <response>
        //    """;

        var query = """
            <instruct>Given a web search query, retrieve relevant passages that answer the query.
            <query>how much protein should a female eat
            <response>
            """;
        var document = """
            As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.
            """;
        var queryEmbedding = pipeline.GenerateEmbeddingFromLastTokenPool(query);
        var documentEmbedding = pipeline.GenerateEmbeddingFromLastTokenPool(document);

        var score = 0f;
        foreach (var (q, d) in queryEmbedding.Zip(documentEmbedding))
        {
            score += q * d * 100;
        }

        Console.WriteLine($"The similarity score between query and document is {score}");
    }
}
