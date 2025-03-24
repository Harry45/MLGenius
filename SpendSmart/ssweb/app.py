"""
Date   : February 2024
Author : Arrykrishna Mootoovaloo
Script : SpendSmart Application
"""

import streamlit as st
from funcs import MONTHS
from funcs import display_all, delete_row_expense_csv
from funcs import add_expenses, add_salary

YEARS = range(2024, 2051, 1)

CAT = [
    "Bills",
    "Groceries",
    "Transportation",
    "Dining Out",
    "Entertainment",
    "Home Maintenance",
    "Healthcare",
    "Education",
    "Savings",
    "Miscellaneous",
]

st.sidebar.header("Inputs")

# form for choosing month and year
form_time = st.sidebar.form("form_time")
form_time.write("Month and Year")
month = form_time.selectbox("Month", MONTHS)
year = form_time.selectbox("Year", YEARS)
submit_time = form_time.form_submit_button()

# form for recording a salary
form = st.sidebar.form("form")
form.write("Do you want to record a salary?")
salary = form.number_input("Salary")
submit = form.form_submit_button()

# form for submitting an expense
form_ex = st.sidebar.form("form_ex")
form_ex.write("Do you want to register an expense?")
submit_no = form_ex.form_submit_button("No")
category = form_ex.selectbox("Categories", CAT)
item = form_ex.text_input("Description of expense")
form_ex.error("Please enter a description of the expense.")
cost = form_ex.number_input("Amount")
date = form_ex.date_input("Expense Date", format="DD.MM.YYYY")
submit_ex = form_ex.form_submit_button()

# form for deleting an expense
form_delete = st.sidebar.form("form_delete")
form_delete.write("Do you want to delete an expense?")
idx = form_delete.number_input("Row Number", step=1, format="%i")
submit_delete = form_delete.form_submit_button("Delete")

st.title(f"SpendSmart App: {month} {year}")

if submit_time:
    display_all(month, year)

if submit:
    add_salary(month, year, salary)
    display_all(month, year)

if submit_ex:
    add_expenses(month, year, date, category, item, cost)
    display_all(month, year)

if submit_delete:
    delete_row_expense_csv(idx, month, year)
    display_all(month, year)
