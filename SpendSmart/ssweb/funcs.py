import os
import streamlit as st
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

EX_COLUMNS = ["Date", "Category", "Item", "Cost"]
SA_COLUMNS = ["Month", "Salary"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def delete_row_expense_csv(row_index: int, month: str, year: int):

    fname = f"expenses/{year}/{month}.csv"
    df = pd.read_csv(fname, index_col=0)
    df = df.drop(row_index)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(fname)


def add_salary(month: str, year: int, salary: float):

    fname = f"salary/{year}.csv"
    if not os.path.isfile(fname):
        os.makedirs(f"salary", exist_ok=True)
        dataframe = pd.DataFrame(columns=SA_COLUMNS)
        dataframe.to_csv(fname, index=True)

    df = pd.read_csv(fname, index_col=0)

    if not month in df["Month"].values:
        new_salary = pd.DataFrame({"Month": month, "Salary": salary}, index=[0])
        df = pd.concat([df if not df.empty else None, new_salary], ignore_index=True)
        df.reset_index(drop=True, inplace=True)

    idx = df[df["Month"].str.contains(month)].index.values
    row = df.iloc[idx]
    current_value = row["Salary"].values[0]
    if salary != current_value:
        df.loc[idx, "Salary"] = salary
    df.to_csv(fname)


def add_expenses(month, year, date, category, item, cost):
    fname = f"expenses/{year}/{month}.csv"
    if not os.path.isfile(fname):
        os.makedirs(f"expenses/{year}", exist_ok=True)
        dataframe = pd.DataFrame(columns=EX_COLUMNS)
        dataframe.to_csv(fname, index=True)

    df_expenses = pd.read_csv(fname, index_col=0)

    if item != "":
        item_d = {"Date": date, "Category": category, "Item": item, "Cost": cost}
        new_df = pd.DataFrame(item_d, index=[0])
        cond = df_expenses if not df_expenses.empty else None
        df = pd.concat([cond, new_df], ignore_index=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(fname)


def display_salary(year: int):
    st.subheader(f"Salary Record")
    df_salary = pd.read_csv(f"salary/{year}.csv", index_col=[0])
    st.write(df_salary)


def display_expenses(month, year):
    following_month = MONTHS[MONTHS.index(month) + 1]
    st.subheader(f"Expenses")
    st.write(f"This corresponds to expenses in {following_month}.")
    df_expenses = pd.read_csv(f"expenses/{year}/{month}.csv", index_col=0)
    st.write(df_expenses)


def display_all(month: str, year: int):
    display_salary(year)
    fname_ex = f"expenses/{year}/{month}.csv"
    if os.path.isfile(fname_ex):
        display_expenses(month, year)
        analysis_pie_expenses(month, year)
    else:
        st.write("No expenses have been recorded yet.")


def analysis_pie_expenses(month: str, year: int):
    df_expenses = pd.read_csv(f"expenses/{year}/{month}.csv", index_col=0)
    if df_expenses.shape[0] >= 1:
        st.subheader(f"Analysis")
        total = df_expenses[["Category", "Cost"]].groupby("Category").sum()
        st.write(total.T)

        data = total["Cost"].values
        labels = list(total.index.values)

        df_salary = pd.read_csv(f"salary/{year}.csv", index_col=[0])
        colors = sns.color_palette("pastel")
        nsalary = df_salary.shape[0]
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].pie(data, labels=labels, colors=colors, autopct="%.0f%%")
        ax[1].bar(
            df_salary["Month"].values,
            df_salary["Salary"],
            width=0.25,
            # color="dodgerblue",
            color=colors[0:nsalary],
        )
        maxn = df_salary["Salary"].values.shape[0]
        ax[1].set_xlim(-0.5, maxn - 0.5)
        ax[1].set_ylabel("Salary in GBP", fontsize=12)
        st.pyplot(fig)

        salary_value = df_salary[df_salary["Month"] == month]["Salary"].values
        total_cost = total["Cost"].sum()
        available = salary_value - total_cost
        st.write(f"Total cost is : GBP {total_cost.item():.2f}")
        st.write(f"Amount available to spend is : GBP {available.item():.2f}")
