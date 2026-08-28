"""
Human-readable label mappings for the UCI Statlog German Credit dataset's
coded categorical values.

Source / documentation: UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc

These mappings are used both by train.py (for EDA readability, optional) and
app.py (so the Streamlit form can show human-readable dropdown labels while
still feeding the model the original coded values it was trained on).
"""

COLUMN_NAMES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "num_dependents",
    "has_telephone",
    "foreign_worker",
    "credit_risk",
]

NUMERIC_COLUMNS = [
    "duration_months",
    "credit_amount",
    "age",
    "installment_rate_pct",
    "residence_since",
    "existing_credits_count",
    "num_dependents",
]

CATEGORICAL_COLUMNS = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "has_telephone",
    "foreign_worker",
]

CHECKING_ACCOUNT_STATUS = {
    "A11": "< 0 DM",
    "A12": "0 <= ... < 200 DM",
    "A13": ">= 200 DM",
    "A14": "no checking account",
}

CREDIT_HISTORY = {
    "A30": "no credits taken / all credits paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now",
    "A33": "delay in paying off in the past",
    "A34": "critical account / other credits existing (not at this bank)",
}

PURPOSE = {
    "A40": "new car",
    "A41": "used car",
    "A42": "furniture / equipment",
    "A43": "radio / television",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "vacation",
    "A48": "retraining",
    "A49": "business",
    "A410": "other",
}

SAVINGS_ACCOUNT = {
    "A61": "< 100 DM",
    "A62": "100 <= ... < 500 DM",
    "A63": "500 <= ... < 1000 DM",
    "A64": ">= 1000 DM",
    "A65": "unknown / no savings account",
}

EMPLOYMENT_SINCE = {
    "A71": "unemployed",
    "A72": "< 1 year",
    "A73": "1 <= ... < 4 years",
    "A74": "4 <= ... < 7 years",
    "A75": ">= 7 years",
}

PERSONAL_STATUS_SEX = {
    "A91": "male : divorced/separated",
    "A92": "female : divorced/separated/married",
    "A93": "male : single",
    "A94": "male : married/widowed",
    "A95": "female : single",
}

OTHER_DEBTORS = {
    "A101": "none",
    "A102": "co-applicant",
    "A103": "guarantor",
}

PROPERTY = {
    "A121": "real estate",
    "A122": "building society savings / life insurance",
    "A123": "car or other (not in savings)",
    "A124": "unknown / no property",
}

OTHER_INSTALLMENT_PLANS = {
    "A141": "bank",
    "A142": "stores",
    "A143": "none",
}

HOUSING = {
    "A151": "rent",
    "A152": "own",
    "A153": "for free",
}

JOB = {
    "A171": "unemployed / unskilled - non-resident",
    "A172": "unskilled - resident",
    "A173": "skilled employee / official",
    "A174": "management / self-employed / highly qualified employee",
}

HAS_TELEPHONE = {
    "A191": "none",
    "A192": "yes, registered under customer's name",
}

FOREIGN_WORKER = {
    "A201": "yes",
    "A202": "no",
}

ALL_MAPPINGS = {
    "checking_account_status": CHECKING_ACCOUNT_STATUS,
    "credit_history": CREDIT_HISTORY,
    "purpose": PURPOSE,
    "savings_account": SAVINGS_ACCOUNT,
    "employment_since": EMPLOYMENT_SINCE,
    "personal_status_sex": PERSONAL_STATUS_SEX,
    "other_debtors": OTHER_DEBTORS,
    "property": PROPERTY,
    "other_installment_plans": OTHER_INSTALLMENT_PLANS,
    "housing": HOUSING,
    "job": JOB,
    "has_telephone": HAS_TELEPHONE,
    "foreign_worker": FOREIGN_WORKER,
}


def reverse_map(mapping: dict) -> dict:
    """Return {label: code} for building dropdowns keyed by human label."""
    return {v: k for k, v in mapping.items()}
