from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import AlbertForMaskedLM
import pkg_resources
import torch
import os
from rxnmapper.tokenization_smiles import SmilesTokenizer
from rxnmapper.smiles_utils import process_reaction


model_path = pkg_resources.resource_filename(
    "rxnmapper", "models/transformers/albert_heads_8_uspto_all_1310k"
)
vocab_path = os.path.join(model_path, "vocab.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()
lazy = {"model": None, "tokenizer": None}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():

    return {"message": "Hello World"}


class InferRequest(BaseModel):
    rxn: str = None
    k: int = 10


def lazy_load():
    if lazy["model"] is None:
        model = AlbertForMaskedLM.from_pretrained(
            model_path,
            output_attentions=False,
            output_past=False,
            output_hidden_states=False,
        )
        lazy["tokenizer"] = SmilesTokenizer(
            vocab_path, max_len=model.config.max_position_embeddings
        )
        model.to(device)
        lazy["model"] = model


@app.post("/api/inference")
def infer(request: InferRequest):
    lazy_load()
    model = lazy["model"]
    tokenizer = lazy["tokenizer"]
    # one_rxn_canon = process_reaction(request.rxn)
    one_rxn_canon = request.rxn
    one_encoded_ids = tokenizer.encode(one_rxn_canon, return_tensors="pt")
    lx = one_encoded_ids.shape[1]
    squared_input = one_encoded_ids.repeat(lx, 1)
    for i in range(0, lx):
        squared_input[i, i] = tokenizer.mask_token_id
    squared_input = squared_input.to(device)
    with torch.no_grad():
        res = model(squared_input)
        logits = res.logits.detach().to("cpu")
        res = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logits_for_masks = torch.stack([logits[i][i] for i, _ in enumerate(logits)])
    prob_for_masks = torch.nn.functional.softmax(logits_for_masks, dim=-1)
    top_k_calc = torch.topk(prob_for_masks, k=request.k)
    t_probs, t_ranks = torch.sort(prob_for_masks, descending=True)
    ranks_correct = [
        (t_ranks[i] == tid).nonzero().tolist()[0][0]
        for i, tid in enumerate(one_encoded_ids[0])
    ]

    return {
        "request": request,
        "results": {
            "topk_probs": top_k_calc.values.numpy().tolist(),
            "topk_tokens": [
                [tokenizer.convert_ids_to_tokens(seq) for seq in sample]
                for sample in top_k_calc.indices.numpy().tolist()
            ],
            "ranks": ranks_correct,
            "rxn_canon": one_rxn_canon,
            "tokens": tokenizer.convert_ids_to_tokens(one_encoded_ids[0]),
        },
    }

