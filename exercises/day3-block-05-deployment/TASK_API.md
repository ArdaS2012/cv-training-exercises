# Adaptation and serving coding reference

The five adjacent TODO contracts define configuration, preprocessing, response construction, contract comparison and (optional stretch) request batching. The Run path consumes the actual functions inside a real localhost POST /predict handler; it reloads the saved model/processor/labels before serving. No hosting/Hub upload occurs.

## APIs
`torch.tensor(data,dtype=torch.float32)` creates a CPU floating tensor; indexing with None introduces an axis. `torch.softmax(logits,dim=1)` normalizes the class axis. `tensor.argmax()` returns the first maximal index; `tensor.tolist()` converts to JSON-compatible Python lists/numbers. `np.allclose(a,b,atol=...,rtol=0)` compares corresponding entries against the absolute bound. Python dict lookup preserves the saved score-index meaning; JSON keys are strings.

`day3.adapt_package(config,epochs=40)` accepts head-only configuration with rate .001–1 and labels 0=even/1=odd. It returns `(report,zip_bytes)` or None for unfinished configuration. With fixed seed123 it resets a TinyCNN backbone/new two-output head, fits 180 shifted training rows for 40 full-batch Adam steps, evaluates 60 fixed validation rows and leaves 60 test rows unused. Shift is clip(.45*x+.25,0,1). Frozen backbone maximum deltas should be zero; changed head weights are not proof of improved validation.

The ZIP contains model.pt, config.json and README.md. Config records processor.shape [8,8], channels1, raw_range [0,16], scale16, string-key id2label, source_revision (SHA256), training settings and architecture. `DomainPipeline.reload(path)` reloads these artifacts. The exercise's custom server reuses its loaded model but routes preprocessing and response creation through student functions. Serving validates finite raw 8×8 grayscale values in [0,16], not normalized or RGB input, and uses a bounded request size. Each temporary localhost server closes after one actual request.

Stretch TODO 5 stacks eight raw validation requests (`day3.day2.target_data()` rows × 16) into one `(8,1,8,8)` tensor; the check runs the frozen Day 2 TinyCNN (`day3.day2.tiny_model()`, ten digit scores) once on the batch and once per request and reports the largest logit difference. `np.stack(list)` adds a new first axis; `tensor.unsqueeze(1)` inserts the channel axis.

Responses contain label:str, scores:two finite Python floats, revision:str. Scores are softmax values in saved output order; ties select first. `same_response` checks label/revision equality and absolute score tolerance. The label-fault drill reuses actual reloaded model/logits and actual request pixels with swapped saved label meanings; the vector stays equal while its semantic result changes. The original package bytes are kept intact.

Training report fields include baseline_validation/adapted_validation dictionaries of source_like/target_shifted accuracy fractions, validation_count, test_count_unused, parameter_max_deltas by tensor name, reload_agreement, config, request, response and package_files. The outer optional trial adds http_response, metadata_fault_response and checks. Downloads are optional and contain computed values without writing fields.

[PyTorch 2.8 tensor](https://docs.pytorch.org/docs/2.8/generated/torch.tensor.html) · [softmax](https://docs.pytorch.org/docs/2.8/generated/torch.softmax.html) · [NumPy allclose](https://numpy.org/doc/2.2/reference/generated/numpy.allclose.html) · [Python HTTP server](https://docs.python.org/3.12/library/http.server.html) · [urllib requests](https://docs.python.org/3.12/library/urllib.request.html).
