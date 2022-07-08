import numpy as np
import torch

from transformers import (
    GPT2Tokenizer,
    OPTForCausalLM,
)


MODEL_NAME = "OPT"


class OPT:
    def __init__(self):
        self.n_gpu = 1
        self.set_seed(42)
        self.max_length = int(10000)
        self.length = 50
        self.stop_token = None
        self.temperature = 1.0
        self.repetition_penalty = 1.0
        self.k = 0
        self.p = 0.9
        self.prefix = ""
        self.padding_text = ""
        self.num_return_sequences = 1
        self.fp16 = True

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def adjust_length_to_model(self, length, max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = (
                max_sequence_length  # No generation bigger than model size
            )
        elif length < 0:
            length = self.max_length  # avoid infinite loop
        return length

    def __call__(self, inputs, length):
        self.length = self.adjust_length_to_model(
            length,
            max_sequence_length=self.model.config.max_position_embeddings,
        )

        prefix = self.prefix if self.prefix else self.padding_text
        encoded_prompt = self.tokenizer.encode(
            prefix + inputs, add_special_tokens=False, return_tensors="pt"
        )
        encoded_prompt = encoded_prompt.to(self.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=self.length + len(encoded_prompt[0]),
            temperature=self.temperature,
            top_k=self.k,
            top_p=self.p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            num_return_sequences=self.num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for _, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(
                generated_sequence, clean_up_tokenization_spaces=True
            )

            # Remove all text after the stop token
            text = text[
                : text.find(self.stop_token) if self.stop_token else None
            ]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                inputs
                + text[
                    len(
                        self.tokenizer.decode(
                            encoded_prompt[0],
                            clean_up_tokenization_spaces=True,
                        )
                    ) :
                ]
            )

            generated_sequences.append(total_sequence)

        return generated_sequences

    def import_data(self, task):
        inputs_list = []
        for item in task["items"]:
            inputs_list.append((item["prompt"], item["length"]))
        return inputs_list

    def import_model(self, device):
        self.device = device
        model_class, tokenizer_class = OPTForCausalLM, GPT2Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained("facebook/opt-2.7b")
        self.model = model_class.from_pretrained("facebook/opt-2.7b")
        self.model.to(self.device)

        if self.fp16:
            self.model.half()
