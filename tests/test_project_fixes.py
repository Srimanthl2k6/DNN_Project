import os
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


class TinyModel(torch.nn.Module):
    def forward(self, x):
        scalar = x.view(x.size(0), -1).mean(dim=1, keepdim=True)
        zeros = torch.zeros(x.size(0), 8, device=x.device, dtype=x.dtype)
        return torch.cat([scalar, -scalar, zeros], dim=1)


def one_hot(index):
    target = torch.zeros(1, 10)
    target[0, index] = 1.0
    return target


class ProjectFixTests(unittest.TestCase):
    def test_visualize_collects_images_by_dataloader_relative_index(self):
        import visualize

        dataloader = [
            (torch.full((2, 3, 2, 2), 1.0), torch.zeros(2, 10)),
            (torch.full((1, 3, 2, 2), 3.0), torch.zeros(1, 10)),
        ]

        selected = visualize.collect_images_by_relative_index(dataloader, [0, 2])

        self.assertEqual([float(image.mean()) for image in selected], [1.0, 3.0])

    def test_visualize_plots_robustness_results_csv(self):
        import visualize

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(
                [
                    {"entropy_group": "high", "attack": "fgsm", "acc_drop": 0.2},
                    {"entropy_group": "high", "attack": "pgd", "acc_drop": 0.3},
                    {"entropy_group": "low", "attack": "fgsm", "acc_drop": 0.4},
                    {"entropy_group": "low", "attack": "pgd", "acc_drop": 0.5},
                ]
            ).to_csv(os.path.join(tmp, "robustness_results.csv"), index=False)

            output = visualize.plot_robustness_drops(tmp)

            self.assertTrue(os.path.exists(output))
            self.assertGreater(os.path.getsize(output), 0)

    def test_fgsm_sweep_returns_grouped_rows_and_filters_indices(self):
        import fgsm_attack

        dataloader = [
            (torch.zeros(1, 3, 4, 4), one_hot(0)),
            (torch.ones(1, 3, 4, 4), one_hot(1)),
            (torch.full((1, 3, 4, 4), 0.5), one_hot(0)),
        ]

        df = fgsm_attack.run_fgsm_epsilon_sweep(
            TinyModel(),
            dataloader,
            {"low": np.array([1]), "high": np.array([2])},
            [0.01],
            torch.device("cpu"),
        )

        self.assertEqual(set(df["entropy_group"]), {"low", "high"})
        self.assertEqual(set(df["epsilon"]), {0.01})
        self.assertEqual(set(df["n_samples"]), {1})
        self.assertEqual(
            list(df.columns),
            [
                "entropy_group",
                "epsilon",
                "n_samples",
                "orig_acc",
                "adv_acc",
                "acc_drop",
                "orig_kl",
                "adv_kl",
                "kl_shift",
                "orig_entropy",
                "adv_entropy",
                "entropy_change",
            ],
        )

    def test_eda_writes_per_class_entropy_and_annotation_certainty(self):
        import data

        probs = np.array(
            [
                [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0],
                [0.1, 0.1, 0.8, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmp:
            np.save(os.path.join(tmp, "cifar10h-counts.npy"), np.rint(probs * 50).astype(int))
            data.eda_visualizations(tmp, probs)

            self.assertTrue(os.path.exists(os.path.join(tmp, "plots", "per_class_entropy.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "plots", "annotation_certainty.png")))

    def test_train_ablation_list_contains_required_experiments(self):
        import train

        names = [entry[0] for entry in train.build_ablations()]

        self.assertEqual(
            names,
            [
                "Exp_KL_Random_Linear",
                "Exp_JS_Random_Linear",
                "Exp_SoftCE_Random_Linear",
                "Exp_CustomDisag_Random_Linear",
                "Exp_KL_Random_MLP",
                "Exp_KL_ImageNet_Linear",
            ],
        )

    def test_evaluate_auto_infers_mlp_imagenet_and_loss_labels(self):
        import evaluate_auto

        self.assertEqual(
            evaluate_auto.infer_experiment_config("Exp_KL_Random_MLP"),
            ("mlp", "random", "KL"),
        )
        self.assertEqual(
            evaluate_auto.infer_experiment_config("Exp_KL_ImageNet_Linear"),
            ("linear", "imagenet", "KL"),
        )
        self.assertEqual(
            evaluate_auto.infer_experiment_config("Exp_SoftCE_Random_Linear"),
            ("linear", "random", "SoftCE"),
        )
        self.assertEqual(
            evaluate_auto.infer_experiment_config("Exp_JS_Random_Linear"),
            ("linear", "random", "JS"),
        )

    def test_best_model_selection_sorts_final_matrix_by_kl(self):
        import best_model_selection

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(
                [
                    {"experiment_name": "slow", "kl": 2.0, "ece": 0.3},
                    {"experiment_name": "best", "kl": 1.0, "ece": 0.4},
                ]
            ).to_csv(os.path.join(tmp, "results.csv"), index=False)

            best_model_selection.select_best_model(tmp)
            out = pd.read_csv(os.path.join(tmp, "FINAL_COMPARISON_MATRIX.csv"))

            self.assertEqual(out.iloc[0]["experiment_name"], "best")

    def test_gradcam_target_layer_uses_backbone_layer4_conv2(self):
        import gradcam_setup
        from model import CustomResNet18

        model = CustomResNet18(head_type="linear", pretrain_strategy="random")

        self.assertIs(gradcam_setup.get_gradcam_target_layer(model), model.backbone[7][-1].conv2)


if __name__ == "__main__":
    unittest.main()
