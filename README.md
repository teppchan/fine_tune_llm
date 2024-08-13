# gemma2-ft

`DayOne` で書き溜めた日記をデータベースにして、小型LLMをFine-Tuningしてみる

## Prepare the environment

```shell
rye sync
```

## Prepare diary data of `DayOne`

First, download your diary data from `DayOne`.
Extract `json` files from the archive file.

```shell
cd src/dayone_prepare
sudo apt install jq
src/dayone_prepare/conv.sh Journal.json
```

You will obtain `out.txt`.

## Fine-tune with `Gemma2 2B`

```shell
rye run python src/gemma2_ft/fine_tune.py
```

## Generate texts with the fine-tuned model

You can generate texts with the original model.
```shell
rye run python src/gemma2_ft/inference.py --text "こんにちは。お元気ですか？"
```

You can generate texts with the fine-tuned model.
```shell
rye run python src/gemma2_ft/inference_ft.py --text "こんにちは。お元気ですか？"
```

## Fine-tune with `Japanese Stable LM 2 1.6B`

```shell
rye run python src/stablelm_ft/fine_tune.py
```

## Generate texts with the fine-tuned model

You can generate texts with the original model.
```shell
rye run python src/stablelm_ft/inference.py --text "こんにちは。お元気ですか？"
```

You can generate texts with the fine-tuned model.
```shell
rye run python src/stablelm_ft/inference_ft.py --text "こんにちは。お元気ですか？"
```


