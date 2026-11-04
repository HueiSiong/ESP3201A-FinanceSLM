"""Local FinanceSLM demonstration. Loads an existing checkpoint; never trains."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
import uuid

from local_runtime import configure

configure()
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch
from tokenizers import Tokenizer

from model import FinanceSLM

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints/final/financeslm_final.pt"
TOKENIZER = ROOT / "data/final/finance_tokenizer.json"
OUTPUT = ROOT / "work/demo"
LOGGER = logging.getLogger("financeslm.demo")


class DemoModel:
    def __init__(self, device=None, checkpoint_path=None, tokenizer_path=None):
        self.checkpoint_path = Path(checkpoint_path or CHECKPOINT)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = ROOT / self.checkpoint_path
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("Model assets are missing. Run python artifacts.py install --from-dir <release-assets>, or fetch --repo OWNER/REPO. See README.md.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lock = threading.Lock()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.tokenizer_path = Path(tokenizer_path or checkpoint.get("tokenizer_path", TOKENIZER))
        if not self.tokenizer_path.is_absolute():
            self.tokenizer_path = ROOT / self.tokenizer_path
        expected_hash = checkpoint.get("tokenizer_sha256")
        if expected_hash and hashlib.sha256(self.tokenizer_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("This checkpoint requires its matching tokenizer; the file checksum differs.")
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        model_class = FinanceSLM
        if checkpoint.get("architecture") == "final_model":
            from final_model import FinanceSLM as FinalFinanceSLM
            model_class = FinalFinanceSLM
        self.model = model_class(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.tokenizer.get_vocab_size() != checkpoint["model_config"]["vocab_size"]:
            raise ValueError("The tokenizer vocabulary does not match the checkpoint.")
        self.model.to(self.device).eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.training_steps = checkpoint.get("training_steps", "unknown")
        self.device_name = (torch.cuda.get_device_name(self.device)
                            if self.device.type == "cuda" else "CPU")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.blocked_ids = [token_id for token in ("<pad>", "<unk>", "<bos>")
                            if (token_id := self.tokenizer.token_to_id(token)) is not None]
        LOGGER.info("Loaded %s on %s (%s parameters)", self.checkpoint_path.name, self.device_name, self.parameters)

    @torch.inference_mode()
    def generate(self, prompt, max_tokens=80, temperature=0.8, top_k=40, seed=42):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a financial sentence to continue.")
        if len(prompt) > 8000:
            raise ValueError("Please shorten your prompt to 8,000 characters or fewer.")
        for label, value, low, high in (("Length", max_tokens, 1, 256),
                                        ("Top-K", top_k, 1, 100),
                                        ("Seed", seed, 0, 2147483647)):
            if value is None or not math.isfinite(float(value)) or int(value) != value or not low <= value <= high:
                raise ValueError(f"{label} must be a whole number between {low} and {high}.")
        if temperature is None or not math.isfinite(float(temperature)) or not 0.1 <= temperature <= 1.5:
            raise ValueError("Temperature must be between 0.1 and 1.5.")
        prompt_ids = self.tokenizer.encode(prompt).ids
        if not prompt_ids:
            raise ValueError("The prompt did not contain any usable tokens.")
        with self.lock:
            context = torch.tensor([prompt_ids[-self.model.max_seq_len:]], dtype=torch.long, device=self.device)
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            new_ids = []
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            for _ in range(int(max_tokens)):
                logits, _ = self.model(context[:, -self.model.max_seq_len:])
                scores = logits[:, -1, :] / float(temperature)
                scores[:, self.blocked_ids] = float("-inf")
                values, indices = torch.topk(scores, k=int(top_k))
                choice = torch.multinomial(torch.softmax(values, dim=-1), 1, generator=generator)
                token_id = indices.gather(1, choice).item()
                if token_id == self.eos_id:
                    break
                new_ids.append(token_id)
                context = torch.cat((context, torch.tensor([[token_id]], device=self.device)), dim=1)
                context = context[:, -self.model.max_seq_len:]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
        continuation = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": self.checkpoint_path.name, "device": self.device_name,
            "tokenizer": str(self.tokenizer_path),
            "prompt": prompt, "continuation": continuation,
            "prompt_tokens": len(prompt_ids), "context_length": self.model.max_seq_len,
            "prompt_truncated": len(prompt_ids) > self.model.max_seq_len,
            "max_new_tokens": int(max_tokens), "temperature": float(temperature),
            "top_k": int(top_k), "seed": int(seed), "generated_tokens": len(new_ids),
            "generation_seconds": elapsed,
            "tokens_per_second": len(new_ids) / elapsed if elapsed else 0,
            "stop_reason": "end_of_document" if len(new_ids) < max_tokens else "length_limit",
        }
        return continuation, record


def build_demo(runtime, model_choices=None):
    model_choices = model_choices or {"Loaded model": str(runtime.checkpoint_path)}
    initial_choice = next((label for label, path in model_choices.items() if (ROOT / path).resolve() == runtime.checkpoint_path.resolve()), next(iter(model_choices)))
    def respond(prompt, max_tokens, temperature, top_k, seed, choice):
        nonlocal runtime
        try:
            chosen = (ROOT / model_choices[choice]).resolve()
            if chosen != runtime.checkpoint_path.resolve():
                runtime = DemoModel(device=str(runtime.device), checkpoint_path=chosen)
            continuation, record = runtime.generate(prompt, max_tokens, temperature, top_k, seed)
            OUTPUT.mkdir(parents=True, exist_ok=True)
            result_file = OUTPUT / f"generation_{uuid.uuid4().hex}.json"
            result_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            LOGGER.info("Generated %s tokens; saved %s", record["generated_tokens"], result_file.name)
            stats = (f"**{record['generated_tokens']} new tokens** · {record['generation_seconds']:.2f} s · "
                     f"{record['tokens_per_second']:.1f} tokens/s · Seed {record['seed']}")
            if record["prompt_truncated"]:
                stats += f"\n\nLong prompt: only its final {runtime.model.max_seq_len} tokens were used as context."
            stats += f"\n\nModel: {runtime.parameters / 1e6:.2f}M parameters; context: {runtime.model.max_seq_len} tokens"
            if not continuation:
                stats += "\n\nThe model ended the passage immediately. Try another prompt or seed."
            return continuation, stats
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except torch.cuda.OutOfMemoryError as error:
            LOGGER.exception("GPU memory exhausted")
            torch.cuda.empty_cache()
            raise gr.Error("GPU memory is full. Close other GPU applications and try again.") from error
        except Exception as error:
            LOGGER.exception("Generation failed")
            raise gr.Error("Generation failed. Details were saved in work/demo/server.log.") from error

    with gr.Blocks(title="FinanceSLM | Financial text studio", analytics_enabled=False) as demo:
        gr.Markdown("# FinanceSLM\n### Financial text, generated locally\n"
                    "Begin a financial report and let a small Transformer continue it.")
        gr.Markdown(f"**Initially loaded · {runtime.parameters / 1e6:.2f}M parameters** &nbsp; | &nbsp; "
                    f"**Ready on {runtime.device_name}** &nbsp; | &nbsp; "
                    f"{runtime.training_steps:,} training steps" if isinstance(runtime.training_steps, int) else
                    f"**Ready on {runtime.device_name}**")
        with gr.Row():
            with gr.Column(scale=5):
                choice = gr.Dropdown(choices=list(model_choices), value=initial_choice, label="Model to run", visible=len(model_choices)>1)
                prompt = gr.Textbox(label="Start your financial passage", lines=5, max_length=8000,
                                    value="The company's revenue increased",
                                    placeholder="For example: Cash flows from operating activities…")
                gr.Examples(examples=[
                    ["The company's revenue increased"],
                    ["Cash flows from operating activities"],
                    ["Operating expenses decreased primarily due to"],
                    ["Management believes the company has sufficient liquidity"],
                ], inputs=prompt, label="Try a starting sentence")
                with gr.Accordion("Generation settings", open=False):
                    length = gr.Slider(1, 256, value=80, step=1, label="Maximum new tokens",
                                       info="Tokens are words, word pieces or punctuation.")
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature",
                                            info="Lower values favour more predictable choices.")
                    top_k = gr.Slider(1, 100, value=40, step=1, label="Top-K",
                                      info="Choose among this many likely next tokens.")
                    seed = gr.Number(value=42, precision=0, minimum=0, maximum=2147483647,
                                     label="Random seed", info="Keep this fixed to repeat an output with the same settings.")
                generate = gr.Button("Generate continuation", variant="primary")
            with gr.Column(scale=6):
                output = gr.Textbox(label="Generated continuation", lines=13, interactive=False, buttons=["copy"],
                                    placeholder="Your model's continuation will appear here.")
                stats = gr.Markdown("Ready. Choose a starting sentence, then generate.")
        gr.Markdown("**Research prototype.** Text may contain invented figures, contradictions or repetition. "
                    "Use it to demonstrate financial writing patterns, not to make financial decisions.\n\n"
                    f"The loaded model uses a rolling context of {runtime.model.max_seq_len} tokens. "
                    "Prompts and outputs are saved locally in `work/demo` for review.")
        generate.click(respond, inputs=[prompt, length, temperature, top_k, seed, choice], outputs=[output, stats],
                       api_name="generate", concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help="Local checkpoint to load; defaults to the validation-selected final model.")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--catalog", type=Path, default=ROOT / "configs/demo_catalog.json", help="JSON mapping model labels to local checkpoints.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(OUTPUT / "server.log", encoding="utf-8"), logging.StreamHandler()])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        runtime = DemoModel(args.device, args.checkpoint, args.tokenizer)
        choices = json.loads(args.catalog.read_text(encoding="utf-8")) if args.catalog else None
        if choices:
            choices = {label: path.replace(chr(92), "/") for label, path in choices.items() if (ROOT / path.replace(chr(92), "/")).is_file()}
        demo = build_demo(runtime, choices)
        demo.launch(server_name="127.0.0.1", server_port=args.port, share=False,
                    inbrowser=args.open_browser, show_error=False,
                    theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["Arial", "sans-serif"]),
                    css=".gradio-container { max-width: 1120px !important; margin: auto; }",
                    footer_links=[])
    except Exception:
        LOGGER.exception("Demo could not start")
        raise


if __name__ == "__main__":
    main()
