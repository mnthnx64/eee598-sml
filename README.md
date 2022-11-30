# EEE598 SML Project

Project repo for EEE598 Statistical Machine Learning. 

## About the Project
The project improves upon the [paper](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_386.pdf) Taking Over the Stock Market: Adversarial Perturbations Against Algorithmic Traders. The paper presents a realistic scenario in which an attacker influences algorithmic trading systems using adversarial learning techniques to manipulate the input data stream in real-time.
The attacker creates a targeted universal adversarial perturbation (TUAP) that is agnostic to the target model and time of use, which remains imperceptible when added to the input stream.

The code uses the data from the [S&P 500 Intraday](https://www.kaggle.com/nickdl/snp-500-intraday-data) dataset and divides it into a set for training the alpha models (Models that algorithmic trading bots use to predict the stock trend), a set for crafting TUAPs, and test sets to evaluate the attack. The training set is used to train three alpha models. Then, we use the TUAP set to craft a universal adversarial perturbation that can fool the target alpha models and evaluate the perturbations` performance.  Finally, we also explore various mitigation methods. Additional information is available in the paper. 

## Setup
Run `setup.sh` to automatically setup the environment and download the data required for training.

## Citation
```
@article{nehemya2021taking,
  title={Taking Over the Stock Market: Adversarial Perturbations Against Algorithmic Traders},
  author={Nehemya, Elior and Mathov, Yael and Shabtai, Asaf and Elovici, Yuval},
  booktitle={ECML-PKDD},
  year={2021}
}
```
