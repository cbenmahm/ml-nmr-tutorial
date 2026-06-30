# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo",
#     "graph-pes==1.0.0",
#     "load-atoms==0.3.10",
#     "e3nn==0.4.4",
#     "torch",
#     "soprano==0.11.1",
#     "matplotlib",
#     "numpy",
#     "ase==3.28.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["ipynb"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess
    from pathlib import Path

    return Path, subprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Atomic tensor models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This has been adapted as a proof of principle from here:
    https://github.com/cbenmahm/graph-pes/blob/update_doc/docs/source/quickstart/atomic-tensor-example.ipynb

    ``graph-pes`` provides models that target atomic "tensorial" properties, ranging from atomic energies and charges, dipoles, NMR anisotropic parameters, to higher rank tensors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/vldgroup/graph-pes/blob/main/docs/source/quickstart/atomic-tensor-example.py)

    Open this notebook in [molab](https://molab.marimo.io), then attach a GPU from the notebook specs button in the app header.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Available models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The currently available models extend the `MACE` and `NequIP` architectures:
    1. [TensorMACE](https://vldgroup.github.io/graph-pes/models/many-body/tensormace.html)

    1. [ZEmbeddingTensorMACE](https://vldgroup.github.io/graph-pes/models/many-body/tensormace.html#graph_pes.models.ZEmbeddingTensorMACE)

    1. [TensorNequIP](https://vldgroup.github.io/graph-pes/models/many-body/tensornequip.html)

    1. [ZEmbeddingTensorNequIP](https://vldgroup.github.io/graph-pes/models/many-body/tensornequip.html#graph_pes.models.ZEmbeddingTensorNequIP)

    Both NequIP- and MACE-based models implement two learning approaches: `direct` and `tensor_product`. To learn tensor components with nonstandard spherical harmonics, such as `0o; 1e; 2o; 3e;...`,  using a MACE-based model, we recommend the tensor_product approach.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generalising to other tensor properties

Although this tutorial focuses on magnetic shielding tensors, the underlying tensor models are completely agnostic to the physical quantity being learned. Any atomic scalar or tensor property that can be decomposed into irreducible representations—,such as atomic energies, electric field gradients (EFGs), or force constants,can be learned using exactly the same workflow. In practice, the only change required is to specify the target irreducible representations (`target_tensor_irreps` in the configuration file); the model architecture and training procedure remain unchanged.

Tensor properties are specified using the **e3nn irreducible representation (irrep) notation**. Each irrep is written as $\ell p$, where:

- $\ell$ is the angular momentum order:
  - `0` = scalar 
  - `1` = vector 
  - `2` = traceless symmetric part of rank-2 tensor 
  - ...

- `p` is the parity:
  - `e` = even (unchanged under spatial inversion)
  - `o` = odd (changes sign under spatial inversion)

Some common examples are:

| Irrep | Physical meaning | Example |
|:------:|------------------|---------|
| `0e` | Scalar | Atomic energy, isotropic shielding |
| `1o` | Polar vector | Force, electric field |
| `1e` | Axial vector (pseudovector) | angular momentum |
| `2e` | Symmetric traceless rank-2 tensor | CSA, EFG |
          """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From a Cartesian tensor to irreps
    A generic rank 2 Cartesian tensor $T$ (of size 3 $\times$ 3), e.g. magnetic shielding tensor, can be decomposed into symmetric and antisymmetric parts:

    - $T_{\mathrm{symm}} = \dfrac{1}{2}(T+T^\mathrm{T})$
    - $T_{\mathrm{antisymm}} = \dfrac{1}{2}(T-T^\mathrm{T})$

    These Cartesian coordinates have their spherical conterparts, that are usually expressed in terms of spherical harmonics (SHs). In this case we obtain the following:

    - a scalar part $\ell=0$: the trace, a rotationally invariant scalar (dimension=1).
    - a pseudovector part $\ell=1$: dual to the antisymmetric part (dimension=3)
    - a quadropole part $\ell=2$: the traceless symmetric part (dimension=5)
          
    The dimensions satisfy $1+3+5=9$, matching the nine Cartesian components of the original tensor.

    In equivariant models, such as `MACE` and `NequIP`, we use features that transform irreducibly under rotations. `e3nn` provides tools to compute exactly these irreducible components from Cartesian tensors, allowing us to use them directly as equivariant features or targets.

    The symmetry-aware conversion between the Cartesian components and the corresponding irreducible spherical representation is handled by the class `e3nn.io.CartesianTensor`.     
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does this mean for magnetic shielding?
    Magnetic shielding tensors are usually symmetric, so the antisymmetric (1o) component is typically zero. That is why the target for magnetic shielding is commonly written as:

    ```yaml
    target_tensor_irreps: "0e + 1e + 2e"
    ```
    
    This means the model is asked to learn the isotropic scalar part (0e), the antisymmetric part (1e), and the traceless symmetric tensor part (2e) of the shielding tensor. Changing this single line allows the same model to learn many other atomic properties with the same overall workflow.
    Changing this single line allows the same model to learn many other atomic properties:
    
    | Property | `target_tensor_irreps` |
    |----------|------------------------|
    | Atomic energy | `"0e"` |
    | Forces | `"1o"` |
    | Electric field gradients (EFG) | `"2e"` |
    | Full magnetic shielding tensor | `"0e + 1o + 2e"` |
    | Symmetric magnetic shielding tensor | `"0e + 2e"` |
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data preparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the remainder of this notebook, we will reconstruct a Tensor NMR model targeting the magnetic shielding tensor (MS) used for amorphous silica in [this paper](https://doi.org/10.1063/5.0274240).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First we download the training data:
    """)
    return


@app.cell
def _(Path):
    from urllib.request import urlretrieve

    train_xyz = Path("train.xyz")
    if not train_xyz.exists():
        urlretrieve(
            "https://github.com/cbenmahm/anistropic-nmr-parameters-data/raw/refs/heads/main/data/train_test/train.xyz",
            train_xyz,
        )
    return (train_xyz,)


@app.cell
def _(train_xyz):
    import load_atoms

    structures = load_atoms.load_dataset(str(train_xyz))
    return (structures,)


@app.cell
def _():
    import graph_pes  # noqa: F401

    return


@app.cell
def _():
    import torch

    # e3nn's cached Wigner constants were saved with an older torch format
    # that is incompatible with torch 2.6+ default weights_only=True loading.
    # Prevent infinite recursion by only patching if not already patched.
    if not hasattr(torch, "_original_load"):
        torch._original_load = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return torch._original_load(*args, **kwargs)

        torch.load = _load

    from e3nn import io

    cartesian_symm = io.CartesianTensor("ij=ji")  # symmetry constraint
    cartesian_antisymm = io.CartesianTensor("ij=-ji")  # antisymmetry constraint
    return cartesian_antisymm, cartesian_symm, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use the two `CartesianTensor` objects to transform the magnetic shielding tensors from Cartesian coordiantes to their irreducible spherical representation, which is the natural way to describe these properties:

    * `cartesian_symm` extracts the symmetric part of the tensor, which consists of the _**scalar**_ part and the _**quadropole**_ part. In terms of `e3nn`'s `Irreps` notations, we obtain the `0e` (scalar) and `2e` (quadropole) terms.

    * `cartesian_antisymm`: extracts the _**pseudovector**_ part of the tensor. For a rank 2 tensor, this corresonds a pseudovector. In terms of `e3nn`'s `Irreps` notations, we obtain the `1o` (pseudovector) term.
    """)
    return


@app.cell
def _(cartesian_antisymm, cartesian_symm, np, structures, torch):
    # 1. Collect shapes so we can split the batch back up afterward
    n_atoms_per_frame = [len(frm) for frm in structures]

    # 2. Stack all (n_atoms, 3, 3) tensors from every frame into one big batch
    all_ms = np.concatenate(
        [frm.arrays["ms"].reshape(-1, 3, 3) for frm in structures], axis=0
    )
    torch_ms = torch.from_numpy(all_ms)  # shape: (tot# 3. Single batched call instead of one call per frame
    symm = cartesian_symm.from_cartesian(torch_ms)
    anti = cartesian_antisymm.from_cartesian(torch_ms)

    # 4. Rearrange to follow the l order: 0, 1, 2 (same as before, just batched)
    ms_all = torch.cat((symm[..., :1], anti, symm[..., 1:]), dim=-1)
    ms_all = ms_all.numpy()

    # 5. Split the batched result back out per frame and assign
    start = 0
    for frm, n in zip(structures, n_atoms_per_frame):
        frm.set_array("tensor" ,ms_all[start : start + n])
        start += n
    return


@app.cell
def _(structures):
    import ase.io

    train, val, test = structures.random_split([0.8, 0.1, 0.1])

    ase.io.write("train-nmr.xyz", train)
    ase.io.write("val-nmr.xyz", val)
    ase.io.write("test-nmr.xyz", test)
    return (test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration file
    """)
    return


@app.cell
def _(mo):
    model_selector = mo.ui.dropdown(
        options={
            "TensorNequIP (Direct)": "nequip",
            "TensorMACE (Tensor Product)": "mace",

        },
        value="TensorMACE (Tensor Product)",
        label="Select Model Architecture",
    )
    model_selector
    return (model_selector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we've saved our labelled structures to suitable files, we're ready to train a model.

    To do this, we have specified the following in the ``tensornequip-direct-sio2.yaml`` file:

    * the model architecture to instantiate and train, here [TensorNequIP](https://vldgroup.github.io/graph-pes/models/many-body/tensornequip.html). Note that we also include a learnable tensor offset component to account for the fact that the amophous silica labels have an arbitrary offset.
    * the data to train on, here a random split of the [amorphous silica](https://github.com/cbenmahm/anistropic-nmr-parameters-data/raw/refs/heads/main/data/train_test/train.xyz) dataset we just downloaded
    * the loss function to use, here a per-atom RMSE
    * and various other training hyperparameters (e.g. the learning rate, batch size, etc.)
    """)
    return


@app.cell(hide_code=True)
def _(mo, model_selector):
    arch_name = "TensorMACE" if model_selector.value == "mace" else "TensorNequIP"
    mo.md(f"""
    The config file is no longer available for download, so we create it inline below. It uses a lightweight ``{arch_name}`` model with a learnable per-element tensor offset, and trains for just a few epochs as a demonstration.
    """)
    return


@app.cell
def _(Path, model_selector):
    config_name = "tensornequip-tp-sio2.yaml" if model_selector.value == "nequip" else "tensormace-tp-sio2.yaml"
    config_path = Path(config_name)

    if model_selector.value == "nequip":
        config_path.write_text(
            "\n".join(
                [
                    "CUTOFF: 5.533",
                    "",
                    "general:",
                    "  progress: rich",
                    "  run_id: train-nequip-tensor",
                    "  log_level: DEBUG",
                    "",
                    "model:",
                    "  offset:",
                    "    +LearnableTensorOffset:",
                    "      length: 9",
                    "  many-body:",
                    "    +TensorNequIP:",
                    "      elements: [\"Si\", \"O\"]",
                    "      cutoff: =/CUTOFF",
                    "      radial_features: 64",
                    "      features:",
                    "        node_irreps: 128x0e+128x1o+128x2e",
                    "        edge_irreps: 0e + 1o + 2e",
                    "      props: tensor",
                    "      layers: 4",
                    "      target_tensor_irreps: 0e + 1o + 2e",
                    "      target_method: direct",
                    "",
                    "data:",
                    "  train:",
                    "    path: train-nmr.xyz",
                    "  valid: val-nmr.xyz",
                    "  test: test-nmr.xyz",
                    "",
                    "loss:",
                    "  - +PropertyLoss:",
                    "      property: tensor",
                    "",
                    "fitting:",
                    "  trainer_kwargs:",
                    "    max_epochs: 200",
                    "    accelerator: auto",
                    "  optimizer:",
                    "    name: AdamW",
                    "    lr: 0.003",
                    "  scheduler:",
                    "    name: ReduceLROnPlateau",
                    "    patience: 25",
                    "    factor: 0.8",
                    "  loader_kwargs:",
                    "    batch_size: 16",
                    "",
                    "wandb: null",
                ]
            )
        )
    else:
        config_path.write_text(
            "\n".join(
                [
                    "CUTOFF: 5.533",
                    "",
                    "general:",
                    "  progress: rich",
                    "  run_id: train-mace-tensor",
                    "  log_level: DEBUG",
                    "",
                    "model:",
                    "  offset:",
                    "    +LearnableTensorOffset:",
                    "      length: 9",
                    "  many-body:",
                    "    +TensorMACE:",
                    "      elements: [\"Si\", \"O\"]",
                    "      cutoff: =/CUTOFF",
                    "      radial_expansion: Bessel",
                    "      n_radial: 8",
                    "      channels: 16",
                    "      hidden_irreps: 128x0e + 128x1o + 128x2e + 128x3o",
                    "      layers: 2",
                    "      target_method: tensor_product",
                    "      target_tensor_irreps: 0e + 1o + 2e",
                    "      number_of_tps: 64",
                    "      irrep_tp: 3o",
                    "      props: tensor",
                    "",
                    "data:",
                    "  train: train-nmr.xyz",
                    "  valid: val-nmr.xyz",
                    "  test: test-nmr.xyz",
                    "",
                    "loss:",
                    "  - +PropertyLoss:",
                    "      property: tensor",
                    "",
                    "fitting:",
                    "  trainer_kwargs:",
                    "    max_epochs: 200",
                    "    accelerator: auto",
                    "  optimizer:",
                    "    name: AdamW",
                    "    lr: 0.01",
                    "  loader_kwargs:",
                    "    batch_size: 16",
                    "",
                    "wandb: null",
                ]
            )
        )
    return (config_path,)


@app.cell(hide_code=True)
def _(config_path, mo):
    mo.md(rf"""
    The training config has been written to ``{config_path}``.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The models are trained in the same way as the usual ``GraphPES`` models using the [graph-pes-train](https://vldgroup.github.io/graph-pes/cli/graph-pes-train/root.html) command. If a trained model already exists, we skip this step so the notebook remains quick to re-run.
    """)
    return


@app.cell
def _(Path, config_path, model_selector, subprocess):
    model_name = "train-nequip-tensor" if model_selector.value == "nequip" else "train-mace-tensor"

    # Robustly find any existing version of this model (with PyTorch Lightning suffixes)
    paths = sorted(Path("graph-pes-results").glob(f"{model_name}*/model.pt"), key=lambda p: p.stat().st_mtime)
    existing_model = paths[-1] if paths else None

    if existing_model:
        model_path = existing_model
    else:
        subprocess.run(
            f"graph-pes-train {config_path} general/run_id={model_name}",
            shell=True,
            check=True,
        )
        # Find the newly created model
        paths = sorted(Path("graph-pes-results").glob(f"{model_name}*/model.pt"), key=lambda p: p.stat().st_mtime)
        model_path = paths[-1] if paths else Path(f"graph-pes-results/{model_name}/model.pt")
    return (model_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we load the model
    """)
    return


@app.cell
def _(model_path, torch):
    from graph_pes.models import load_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = load_model(str(model_path)).to(device).eval()  # load the model  # move to GPU if available  # set to evaluation mode
    return (best_model,)


@app.cell
def _(model_path):
    model_path
    return


@app.cell
def _(best_model):
    from graph_pes.utils.calculator import GraphPESCalculator

    calculator = GraphPESCalculator(best_model)
    return (calculator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then, we need to tranform the predictions back to Cartesian coordinates
    """)
    return


@app.cell
def _(calculator, cartesian_antisymm, cartesian_symm, test, torch):
    for frm_1 in test:
        calculator.calculate(frm_1, properties=['tensor'])
        tensor = calculator.results['tensor']
        tensor = torch.from_numpy(tensor)
        symm_1 = torch.cat((tensor[..., :1], tensor[..., 4:]), dim=-1)
        tensor_symm = cartesian_symm.to_cartesian(symm_1)
        tensor_antisymm = cartesian_antisymm.to_cartesian(tensor[..., 1:4])
        tensor = tensor_symm + tensor_antisymm
        tensor = tensor.cpu().numpy()
        frm_1.arrays['ms_ML'] = tensor
    predictions_completed = True
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    and now you can use libraries like [soprano](https://github.com/CCP-NC/soprano) to exctract NMR tensor properties or [MRSimulator](https://github.com/deepanshs/mrsimulator) to simulate spectra!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In what follows, we compare the magnetic shielding isotropy obtained from DFT and ML calculations. We also examine one quantity that characterizes the anisotropy of the shielding tensor: the asymmetry parameter $\zeta$.

    To extract all of these values, we will use the ``soprano`` package.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: soprano !pip install soprano

    import numpy as np
    from soprano.properties import nmr

    return nmr, np


@app.cell
def _(test):
    # soprano expects the tensors to have the shape (n_atoms, 3, 3)
    # Guard ms reshape: only reshape if not already 3-D (avoids double-reshape on model switch)
    for frm_2 in test:
        if frm_2.arrays['ms'].ndim == 2:
            frm_2.arrays['ms'] = frm_2.arrays['ms'].reshape(-1, 3, 3)
        frm_2.arrays['ms_ML'] = frm_2.arrays['ms_ML'].reshape(-1, 3, 3)
    reshaping_completed = True
    return


@app.cell
def _(nmr, np, test):

    shielding_dft = np.concatenate(
        [nmr.MSIsotropy.get(frame, tag="ms") for frame in test]
    )

    shielding_ml = np.concatenate(
        [nmr.MSIsotropy.get(frame, tag="ms_ML") for frame in test]
    )
    return shielding_dft, shielding_ml


@app.cell
def _():
    import matplotlib.pyplot as plt
    from soprano.selection import AtomSelection

    return AtomSelection, plt


@app.cell
def _(AtomSelection, model_path, np, plt, shielding_dft, shielding_ml, test):
    print('model: ', model_path)
    Si_inds = np.concatenate([AtomSelection.from_element(frame, 'Si').indices for frame in test])
    plt.scatter(shielding_dft[Si_inds], shielding_ml[Si_inds], color="crimson", s=4)

    plt.axline((shielding_dft[Si_inds][0], shielding_dft[Si_inds][0]),slope=1 , lw=1, color="k")
    plt.xlabel("DFT isotropic shielding (ppm)")
    plt.ylabel("ML isotropic shielding (ppm)")
    plt.gcf()
    return (Si_inds,)


@app.cell
def _(nmr, np, test):
    asymmetry_dft = np.concatenate(
        [nmr.MSAsymmetry.get(frame, tag="ms") for frame in test]
    )

    asymmetry_ml = np.concatenate(
        [nmr.MSAsymmetry.get(frame, tag="ms_ML") for frame in test]
    )
    return asymmetry_dft, asymmetry_ml


@app.cell
def _(Si_inds, asymmetry_dft, asymmetry_ml, plt):
    plt.scatter(asymmetry_dft[Si_inds], asymmetry_ml[Si_inds], color="crimson", s=4)

    plt.axline((asymmetry_dft[Si_inds][0], asymmetry_dft[Si_inds][0]), slope=1, lw=1, color="k")
    plt.xlabel("DFT asymmetry")
    plt.ylabel("ML asymmetry")
    plt.gcf()
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 1: Test on unseen structures

    In the main tutorial, the model is evaluated on a random split of the same amorphous silica dataset used for training. A more demanding test is to apply the trained model to **unseen structures** that are chemically related but structurally different, such as large amorphous SiO\(_2\) models or zeolite frameworks.

    This kind of test is useful because it probes whether the model is learning a genuinely transferable local environment representation, rather than only interpolating within the training distribution.

    In the next cells, you can download an unseen test set. Evaluate the same trained model on it!
    """)
    return

@app.cell
def _(Path):
    # Placeholder for an unseen test dataset.
    # Replace the URL below with the dataset you want to use for the tutorial.
    from urllib.request import urlretrieve

    unseen_xyz = Path("unseen_test.xyz")
    if not unseen_xyz.exists():
        urlretrieve(
            "https://github.com/cbenmahm/anistropic-nmr-parameters-data/raw/refs/heads/main/data/aSiO2_models/aSiO2_fast_ml_qm_ms_efg.xyz",
            unseen_xyz,
        )
    return (unseen_xyz,)

@app.cell

def _():
    # write your code here
    ...

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Challenge 2: Fine-tune a pretrained model on zeolites

    Another useful workflow is **fine-tuning**. Instead of training a model from scratch on a smaller zeolite dataset, we can start from the silica model trained above and fine-tune it on the new data.

    This is often a practical strategy when the new dataset is limited in size or spans a narrower chemical space. The idea is to reuse the local chemical knowledge already learned by the pretrained model, then adapt it to the zeolite domain.

    The next cells provide the zeolite dataset.
    
    Two minimal change need to be made to the config file:
    
    1. Instead of defining the model hyperparameters, we provide the path to the already pre-trained model:
    ```yaml
          model:
            +load_model_component:
                path: path/to/model.pt
                key: many-body
    ```
    1. Repalce the paths of the previous training/validation sets with the new ones
          
    You might find useful to reduce the learning rate as well.
    """)
    return


@app.cell
def _(Path):
    # Placeholder for a zeolite fine-tuning dataset.
    # Replace the URL below with the dataset you want to use for the tutorial.
    from urllib.request import urlretrieve

    zeolite_xyz = Path("zeolite_train.xyz")
    if not zeolite_xyz.exists():
        urlretrieve(
            "https://github.com/cbenmahm/anistropic-nmr-parameters-data/raw/refs/heads/main/data/hypothetical_zeolites/hypozeo_ml_dft_isd_tp_ms_efg.xyz",
            zeolite_xyz,
        )
    return (zeolite_xyz,)

@app.cell
def _():
    # write your code here
    ...

if __name__ == "__main__":
    app.run()
