using AutoGen.Core;
using System.Text.Json.Serialization;
using System.Text.Json;

namespace Local.NET.LLAMA;

internal class LlamaFunctionCall
{
    /// <summary>
    /// get weather from city
    /// </summary>
    /// <param name="city"></param>
    [Function]
    public Task<string> GetWeather(string city)
    {
        return Task.FromResult($"The weather in {city} is sunny.");
    }

    private class GetWeatherSchema
    {
        [JsonPropertyName(@"city")]
        public required string City { get; set; }
    }

    public Task<string> GetWeatherWrapper(string arguments)
    {
        var schema = JsonSerializer.Deserialize<GetWeatherSchema>(
            arguments,
            new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            });

        return GetWeather(schema!.City);
    }

    public FunctionContract GetWeatherFunctionContract
    {
        get => new FunctionContract
        {
            Namespace = @"Local.NET.Llama",
            ClassName = @"MistralFunctionCall",
            Name = @"GetWeather",
            Description = @"get weather from city",
            ReturnType = typeof(Task<string>),
            Parameters =
            [
                    new FunctionParameterContract
                    {
                        Name = @"city",
                        Description = @"city. type is string",
                        ParameterType = typeof(string),
                        IsRequired = true,
                    },
            ],
        };
    }
}
