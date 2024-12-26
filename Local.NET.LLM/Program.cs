using Local.NET.Mistral;

var weightFolder = @"C:\Users\maxim\source\repos\models\mistral\Mistral-7B-Instruct-v0.3";

await new Mistral(weightFolder).ChatAsync();
await new Mistral(weightFolder).WeatherChatAsync();
Mistral.Embedding();