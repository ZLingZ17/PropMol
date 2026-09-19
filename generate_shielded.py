import argparse, os, sys
import json
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(BASE_DIR, 'checkpoint_final.ckpt')
VOCAB_FILE = os.path.join(BASE_DIR, 'vocab_bpe_300.txt')
BERT_CONFIG_TEXT = os.path.join(BASE_DIR, 'config_bert.json')
BERT_CONFIG_PROPERTY = os.path.join(BASE_DIR, 'config_bert_property.json')
NORMALIZE_PKL = os.path.join(BASE_DIR, 'train_all/normalize_1.pkl')
PROPERTY_NAME_FILE = os.path.join(BASE_DIR, 'property_name.txt')
CALC_PROPERTY_FILE = os.path.join(BASE_DIR, 'calc_property.py')

EMBED_DIM = 256
DEVICE = 'cuda'


def get_property_names():
    # 读取所有性质名称
    with open(PROPERTY_NAME_FILE, 'r') as f:
        return [n.strip() for n in f.readlines()]


def update_shield_list(shield_props):
    with open(CALC_PROPERTY_FILE, 'r') as f:
        content = f.read()

    new_list_str = str(shield_props) if shield_props else '[]'

    new_content = re.sub(
        r"^shield_list\s*=\s*\[[^\]]*\]",
        f"shield_list = {new_list_str}",
        content,
        flags=re.MULTILINE,
    )

    if new_content == content:
        print("Warning: shield_list line not found or unchanged in calc_property.py")
        return False

    with open(CALC_PROPERTY_FILE, 'w') as f:
        f.write(new_content)
    print(f"Updated shield_list in calc_property.py -> {new_list_str}")
    return True


def resolve_shield_props(properties):
    all_names = get_property_names()

    if not properties:
        return []

    invalid = [p for p in properties if p not in all_names]
    if invalid:
        print(f"Error: Invalid property names: {invalid}")
        print("Available properties:")
        for i, name in enumerate(all_names):
            print(f"  {i:>2}: {name}")
        sys.exit(1)

    return list(properties)


def update_shield(shield_props):
    update_shield_list(shield_props)

    all_names = get_property_names()
    shield_idx_set = set()
    for p in shield_props:
        if p in all_names:
            shield_idx_set.add(all_names.index(p))

    return shield_props, shield_idx_set


def generate(path, config_file):
    with open(config_file, 'r') as f:
        cfg = json.load(f)

    input_file = cfg.get('input_file', '')
    k = cfg.get('k', 2)
    n_generate = cfg.get('n_generate', 1)
    properties = cfg.get('properties', [])

    if not input_file:
        print("Error: 'input_file' is required in config json")
        sys.exit(1)

    shield_props = resolve_shield_props(properties)
    shield_props, shield_idx_set = update_shield(shield_props)

    import torch
    import numpy as np
    import random
    import pickle
    from tqdm import tqdm
    from rdkit import Chem
    import torch.backends.cudnn as cudnn
    from transformers import BertTokenizer, WordpieceTokenizer
    from torch.utils.data import DataLoader
    from PropMol_models import PropMol
    from dataset import SMILESDataset_pretrain
    from calc_property import calculate_property
    from d_pv2smiles_single import generate as gen_step, BinarySearch

    model_config = {
        'embed_dim': EMBED_DIM,
        'bert_config_text': BERT_CONFIG_TEXT,
        'bert_config_property': BERT_CONFIG_PROPERTY,
    }
    device = torch.device(DEVICE)

    seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 5. Dataset
    print("Creating dataset")
    dataset_test = SMILESDataset_pretrain(input_file, data_length=None)
    test_loader = DataLoader(dataset_test, batch_size=1, pin_memory=True, drop_last=False)

    # 6. Tokenizer
    tokenizer = BertTokenizer(
        vocab_file=VOCAB_FILE,
        do_lower_case=False,
        do_basic_tokenize=False,
    )
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab,
        unk_token=tokenizer.unk_token,
        max_input_chars_per_word=250,
    )

    # 7. Model
    print("Creating model")
    model = PropMol(config=model_config, tokenizer=tokenizer, no_train=True)

    print('LOADING PRETRAINED MODEL..')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']

    for key in list(state_dict.keys()):
        if 'word_embeddings' in key and 'property_encoder' in key:
            del state_dict[key]
        if 'queue' in key:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print(f'load checkpoint from {CHECKPOINT}')
    print(msg)
    model = model.to(device)

    stochastic = True  

    print("=" * 50)
    print(f"PV-to-SMILES generation (stochastic={stochastic}, k={k}, n_generate={n_generate})...")
    model.eval()
    reference, candidate = [], []
    for (prop, text) in tqdm(test_loader):
        prop = prop.to(device, non_blocking=True)
        property1 = model.property_embed(prop.unsqueeze(2))
        properties = torch.cat([model.property_cls.expand(property1.size(0), -1, -1), property1], dim=1)
        prop_embeds = model.property_encoder(inputs_embeds=properties, return_dict=True).last_hidden_state

        for _ in range(n_generate):
            product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(device)
            values, indices = gen_step(model, prop_embeds, product_input, stochastic=stochastic, k=k)
            product_input = torch.cat(
                [torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(device), indices.squeeze(0).unsqueeze(-1)],
                dim=-1,
            )
            current_p = values.squeeze(0)
            final_output = []
            for _ in range(100):
                values, indices = gen_step(model, prop_embeds, product_input, stochastic=stochastic, k=k)
                k2_p = current_p[:, None] + values
                product_input_k2 = torch.cat(
                    [product_input.unsqueeze(1).repeat(1, k, 1), indices.unsqueeze(-1)], dim=-1
                )
                if tokenizer.sep_token_id in indices:
                    ends = (indices == tokenizer.sep_token_id).nonzero(as_tuple=False)
                    for e in ends:
                        p = k2_p[e[0], e[1]].cpu().item()
                        final_output.append((p, product_input_k2[e[0], e[1]]))
                        k2_p[e[0], e[1]] = -1e5
                    if len(final_output) >= k ** 1:
                        break
                current_p, i = torch.topk(k2_p.flatten(), k)
                next_indices = torch.from_numpy(
                    np.array(np.unravel_index(i.cpu().numpy(), k2_p.shape))
                ).T
                product_input = torch.stack(
                    [product_input_k2[i[0], i[1]] for i in next_indices], dim=0
                )

            reference.append(text[0].replace('[CLS]', ''))
            candidate_k = []
            final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]
            for p, sentence in final_output:
                cdd = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(sentence[:-1])
                ).replace('[CLS]', '')
                candidate_k.append(cdd)
            if candidate_k:
                candidate.append(random.choice(candidate_k))
            else:
                candidate.append('')

    with open(NORMALIZE_PKL, 'rb') as w:
        norm = pickle.load(w)
    property_mean, property_std = norm

    from rdkit import Chem
    from rdkit.Chem import Draw

    all_names = get_property_names()
    valids = []
    n_mse = []
    generate_list = []  

    for i in range(len(reference)):
        try:
            prop_ref = calculate_property(reference[i])
            prop_cdd = calculate_property(candidate[i])
            n_ref = (prop_ref - property_mean) / property_std
            n_cdd = (prop_cdd - property_mean) / property_std
            n_mse.append((n_ref - n_cdd) ** 2)
            valids.append(candidate[i])

            mol = Chem.MolFromSmiles(candidate[i])
            if mol is None:
                continue
            prop_values = calculate_property(candidate[i]).tolist()

            img_name = f'mol_{i}.png'
            img_path = os.path.join(path, img_name)
            img = Draw.MolToImage(mol, size=(300, 300))
            img.save(img_path)

            mol_info = {'smiles': candidate[i], 'image': img_name}
            for idx, name in enumerate(all_names):
                mol_info[name] = f"{prop_values[idx]:.6f}"
            generate_list.append(mol_info)
        except Exception as e:
            print(f"Skipping invalid molecule {i}: {e}")
            continue

    result_data = {}
    result_data['Shielded properties'] = str(shield_props) if shield_props else 'None'

    if len(n_mse) != 0:
        n_mse = torch.stack(n_mse, dim=0)
        n_rmse = torch.sqrt(torch.mean(n_mse, dim=0))
        result_data['Mean nRMSE (all 56 properties)'] = f"{n_rmse.mean().item():.6f}"

        if shield_idx_set:
            shielded_rmse = n_rmse[list(shield_idx_set)]
            non_shield_idx = [j for j in range(len(all_names)) if j not in shield_idx_set]
            if non_shield_idx:
                non_shielded_rmse = n_rmse[non_shield_idx]
                result_data[f'Mean nRMSE (non-shielded, {len(non_shield_idx)} props)'] = f"{non_shielded_rmse.mean().item():.6f}"
            result_data[f'Mean nRMSE (shielded, {len(shield_idx_set)} props)'] = f"{shielded_rmse.mean().item():.6f}"
    else:
        result_data['Mean nRMSE (all 56 properties)'] = "No valid molecules"

    v = len(valids)
    result_data['Validity'] = f"{v / len(candidate):.4f}" if len(candidate) > 0 else "0"

    canon_lines = [Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False) for l in valids]
    unique_lines = list(set(canon_lines))
    u = len(unique_lines)
    result_data['Uniqueness'] = f"{u / v:.4f}" if v > 0 else "0"

    result_data['generate'] = generate_list

    smiles_json_path = os.path.join(path, 'smiles.json')
    with open(smiles_json_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved in '{smiles_json_path}'")
    print(f"Generated {len(generate_list)} valid molecules with images and 56 properties.")
    print("=" * 50)


def main(path, config_file):
    if not os.path.exists(path):
        os.makedirs(path)

    generate(path, config_file)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
