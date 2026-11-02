# STRATUM

### Intelligent Virtual Metrology and Multi-Parameter Quality Monitoring for Semiconductor Manufacturing

STRATUM is a semiconductor manufacturing analytics platform designed to estimate post-deposition material properties from manufacturing process data and support multi-parameter process quality monitoring.

## Problem Statement

Semiconductor Physical Vapor Deposition (PVD) produces metal layers whose physical and electrical properties must be monitored.

STRATUM aims to support semiconductor engineers through machine-learning-based virtual metrology, statistical analysis, automated quality monitoring, and evidence-grounded engineering reports.

## Planned Features

- Predict AlCu metal-layer mean thickness using process sensor data.
- Investigate sheet-resistance prediction as a second regression task.
- Analyse 17-point wafer thickness and electrical measurements.
- Monitor quality indicators and process stability.
- Compare equipment-level manufacturing trends.
- Automate dataset validation, prediction, and reporting.
- Generate evidence-grounded engineering summaries using an LLM.
- Provide an interactive web dashboard.

## Dataset

**Source:** Infineon Technologies semiconductor manufacturing dataset

**Dataset:** Advanced Process Control and Statistical Process Control Data for Thickness Prediction of AlCu and WTi Metal Layers in Semiconductor Manufacturing

**Version:** 2 (October 2026)

**Source URL:** https://zenodo.org/records/23442538

**AlCu records:** 5,208

**Columns:** 178

The original CSV is excluded from Git. Download it from Zenodo and place it in:

`data/raw/apc_spc_AlCu_inputs_outputs.csv`

## Planned Technology Stack

- Python 3.14
- Pandas, NumPy, Scikit-learn
- React
- FastAPI
- PostgreSQL
- LLM API integration
- Git and GitHub

## Current Development Status

- [x] Project folder structure
- [x] Python virtual environment
- [x] Initial dataset exploration
- [ ] Data preprocessing and detailed EDA
- [ ] Machine learning model development
- [ ] Backend and database
- [ ] Frontend
- [ ] Automation and LLM integration
- [ ] Testing and deployment

## Project Type

Individual final-year minor project (3 credits).

## Reference

Mevic, A., Laber, A., & Krivic, S. (2025).
Application of Multi-Output Regression and Feature Selection Methods in Semiconductor Manufacturing.
ECAI 2025.

https://doi.org/10.3233/FAIA251482