using System.Collections.Immutable;
using System.Text;
using LLama.Batched;
using LLama.Common;
using LLama.Native;
using Spectre.Console;
using LLama.Sampling;

namespace LLama.Examples.Examples;

public class BatchedExecutorBoolQ
{
    // Answers may start with a space, and then must produce one of the listed strings followed by a newline character and nothing else.
    private static readonly Grammar AnswerGrammar =
        new("""
            root ::= answer whitespace?

            answer ::= "true" | "false" | "yes" | "no"

            whitespace ::= [ \t\n\r]+
            """, "root");

    public static async Task Run()
    {
        // Load model weights
        var parameters = new ModelParams(UserSettings.GetModelPath());
        
        using var model = await LLamaWeights.LoadFromFileAsync(parameters);

        var requestedMaxActive = AnsiConsole.Ask("How many parallel conversations to evaluate in a batch", 64);
        var sys = AnsiConsole.Ask("System prompt", "Answer the question with a single word answer.");

        // Create an executor that can evaluate a batch of conversations together
        using var executor = new BatchedExecutor(model, parameters);

        // Print some info
        var name = model.Metadata.GetValueOrDefault("general.name", "unknown model name");
        Console.WriteLine($"Created executor with model: {name}");

        // Load dataset
        var data = new List<(string, bool, string)>();
        if (AnsiConsole.Ask("Load training dataset?", false))
            data.AddRange(LoadData("Assets/BoolQ/train.csv"));
        if (AnsiConsole.Ask("Load validation dataset?", true))
            data.AddRange(LoadData("Assets/BoolQ/validation.csv"));
        AnsiConsole.MarkupLineInterpolated($"Loaded Dataset: {data.Count} questions");
        var limit = AnsiConsole.Ask("Limit dataset size", 1000);
        if (data.Count > limit)
            data = data.Take(limit).ToList();

        var queue = new Queue<(string, bool, string)>(data);

        // Dynamic management of max active conversations based on KV slot availability
        var currentMaxActiveSize = requestedMaxActive; // Start with maximum of requestedMaxActive, but will reduce if we hit NoKvSlot errors
        var activeConversations = ImmutableList.Create<ConversationRunner>();
        
        // storing the number of times the model gets the question right, with 
        // separate counters for when the answer was true vs false
        int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;

        NativeLogConfig.llama_log_set((level, message) => {});

        await AnsiConsole.Live(new Table()).StartAsync(async ctx =>
        {
            while (queue.Count > 0 || activeConversations.Any(d => !d.IsFinished))
            {
                // Only add new conversations if we haven't reached KV limits or if we have no active conversations
                if (queue.Count > 0 && activeConversations.Count < currentMaxActiveSize)
                {
                    for (var i = activeConversations.Count; i < currentMaxActiveSize; i++)
                    {
                        if (!queue.TryDequeue(out var runner)) continue;
                        var (question, answer, hint) = runner;
                        var conversationData = new ConversationRunner(executor, sys, question, answer, hint);
                        activeConversations = activeConversations.Add(conversationData);
                    }
                }
                

                var decodeResult = await executor.Infer();

                if (decodeResult == DecodeResult.NoKvSlot)
                {
                    // Reduce the maxActive limit to prevent hitting the limit again
                    // Ensure we don't go below 1 as we need at least one conversation
                    currentMaxActiveSize = Math.Max(1, activeConversations.Count - 1);
                    
                    // Handle the NoKvSlot error by selecting a conversation to dispose
                    if (activeConversations.Count > 0)
                    {
                        var conversationToDispose = activeConversations
                            .OrderByDescending(c => c.TokensGenerated)
                            .First();

                        // Get the original prompt to add back to the queue
                        var promptToRequeue = (conversationToDispose.Question, conversationToDispose.Answer,
                            conversationToDispose.Hint);

                        // Add it back to the queue
                        queue.Enqueue(promptToRequeue);

                        // Dispose the conversation
                        conversationToDispose.Dispose();

                        // Remove from active conversations list
                        activeConversations = activeConversations.Remove(conversationToDispose);

                        // Continue to next iteration to process the modified active conversations
                        continue;
                    }
                }
                else if (decodeResult == DecodeResult.Error)
                {
                    throw new Exception("Unknown error occurred while inferring.");
                }

                foreach (var conversationData in activeConversations.Where(conversationData => !conversationData.IsFinished))
                {
                    conversationData.Sample();
                    conversationData.Prompt();
                    ctx.UpdateTarget(BuildStatusTable(name, executor, queue, activeConversations, truePositive, trueNegative, falsePositive, falseNegative));
                }

                if (activeConversations.Any(i => i.IsFinished))
                {
                    foreach (var item in activeConversations.Where(i => i.IsFinished))
                    {
                        item.Result(ref truePositive, ref trueNegative, ref falsePositive, ref falseNegative);
                        item.Dispose();
                    }

                    // remove all completed conversations from the list.
                    activeConversations = activeConversations.RemoveAll(i => i.IsFinished);
                }
                
                ctx.UpdateTarget(BuildStatusTable(name, executor, queue, activeConversations, truePositive, trueNegative, falsePositive, falseNegative));
            }
        });
    }

    private static LLamaKvCacheViewSafeHandle? _debugView;
    private static Table BuildStatusTable(string model, BatchedExecutor executor, Queue<(string, bool, string)> queue, ImmutableList<ConversationRunner> activeConversations, int tp, int tn, int fp, int fn)
    {
        if (_debugView == null)
        {
            _debugView = executor.Context.NativeHandle.KvCacheGetDebugView();
        }
        else
        {
            _debugView.Update();
        }

        return new Table()
                
            .AddColumns("Key", "Value")
            .AddRow("Model", model)
            .AddRow("Remaining Questions", queue.Count.ToString())
            .AddRow("Active Conversations", activeConversations.Count.ToString())
            .AddRow("Correct", $"[green]{tp + fp}[/]")
            .AddRow("Incorrect", $"[red]{tn + fn}[/]")
            .AddRow("KV Cache Cells (Used/Total)", _debugView.UsedCellCount + "/" + _debugView.CellCount )
            ;
    }

    private static IEnumerable<(string, bool, string)> LoadData(string path)
    {
        foreach (var line in File.ReadLines(path))
        {
            var splits = line.Split(",");

            if (!bool.TryParse(splits[1], out var boolean))
                continue;

            var hint = string.Join(",", splits[2..]);
            hint = hint.Trim('\"');

            yield return (splits[0], boolean, hint);
        }
    }

    /// <summary>
    /// All the mechanics necessary to run a conversation to answer a single question
    /// </summary>
    private class ConversationRunner
        : IDisposable
    {
        private readonly BatchedExecutor _executor;
        private readonly StreamingTokenDecoder _decoder;
        private readonly ISamplingPipeline _sampler;

        private readonly Conversation _conversation;
        private LLamaToken? _sampledToken;

        public string Question { get; }
        public bool Answer { get; }
        public string Hint { get; }
        public bool IsFinished { get; private set; }

        public int TokensGenerated { get; private set; }

        public ConversationRunner(BatchedExecutor executor, string sys, string question, bool answer, string hint)
        {
            _executor = executor;
            _decoder = new StreamingTokenDecoder(executor.Context);
            _sampler = new DefaultSamplingPipeline
            {
                Grammar = AnswerGrammar, 
                GrammarOptimization = DefaultSamplingPipeline.GrammarOptimizationMode.Extended
            };

            // Make sure question ends with question mark
            if (!question.EndsWith('?'))
                question += '?';


            Question = question;
            Answer = answer;
            Hint = hint;

            if (!hint.EndsWith('.'))
                hint += '.';
            question = $"{hint}\n{question}\nAnswer with Yes or No based .";

            // Template the question
            var template = new LLamaTemplate(executor.Model);
            template.Add("system", sys);
            template.Add("user", question);
            template.AddAssistant = true;
            var templatedQuestion = Encoding.UTF8.GetString(template.Apply());

            // Prompt
            _conversation = executor.Create();
            var tokenizedPrompt = _executor.Context.Tokenize(templatedQuestion, special: true, addBos: true);
            TokensGenerated += tokenizedPrompt.Length;
            _conversation.Prompt(tokenizedPrompt);
        }

        public void Sample()
        {
            if (IsFinished)
                return;
            if (!_conversation.RequiresSampling)
                return;

            var token = _conversation.Sample(_sampler);

            TokensGenerated++;

            var vocab = _executor.Context.Vocab;
            if (token.IsEndOfGeneration(vocab) || vocab.Newline == token)
            {
                _sampledToken = default;
                IsFinished = true;
            }
            else
            {
                _sampledToken = token;
            }
        }

        public void Prompt()
        {
            if (IsFinished)
                return;
            if (!_sampledToken.HasValue)
                return;

            var token = _sampledToken.Value;
            _sampledToken = default;

            _decoder.Add(token);
            _conversation.Prompt(token);
        }

        public void Result(ref int tp, ref int tn, ref int fp, ref int fn)
        {
            var str = _decoder.Read().Trim();
            var result = str switch
            {
                "true" or "yes" => true,
                _ => false,
            };

            switch (Answer, result)
            {
                case (true, true): tp++; break;
                case (true, false): fn++; break;
                case (false, true): fp++; break;
                case (false, false): tn++; break;
            }
        }

        public void Dispose()
        {
            _conversation.Dispose();
            _sampler.Dispose();
        }
    }
}