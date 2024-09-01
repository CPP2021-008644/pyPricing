import shelve, sys, os, shutil, json
from datetime import datetime
from pathlib import Path
import numpy as np
import time
import math
import io
import requests

import pandas as pd

ROOT = Path(__file__).resolve().parent


def readDataPath():
    data_path = ROOT / "dummydata"
    return str(data_path)


letter_month_dict = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}


def compute_generic_asset(generic_ticker, date=None, start_date=None, end_date=None):
    tenor = generic_ticker[-2]
    distance = int(generic_ticker[-1])
    if date is not None:
        date = pd.to_datetime(date)
        if tenor == "M":
            letter_month = letter_month_dict[
                (date.month + distance) % 12 + 12 * (date.month + distance == 12)
            ]
            year = date.year + (date.month == 12)
            ticker = generic_ticker[:-1] + letter_month + str(year)[-2:]
        # elif tenor == "Q":

        else:
            raise ValueError("Tenor letter not known.")
        return ticker
    else:
        assert start_date is not None and end_date is not None, (
            "No dates were provided."
        )
        df = pd.DataFrame(index=pd.date_range(start_date, end_date), columns=["ticker"])
        for date in df.index:
            df.loc[date, "ticker"] = compute_generic_asset(generic_ticker, date=date)
        result = df.reset_index().groupby("ticker").min()
        result = result.rename(columns={result.columns[0]: "start_date"})
        result["end_date"] = df.reset_index().groupby("ticker").max()
        return result


class BeautifulDataAFSStyle:
    def __init__(
        self,
        db_path=readDataPath() + "/DB/HIST/",
        remote_api_base_url: str = "http://10.66.40.13:15555/api/v1/s3",
        remote_prefix: str = "DB/HIST",
    ):
        if db_path[-1] != "/":
            db_path = db_path + "/"
        self.db_path = db_path
        self.main = self.db_path + "MAIN/"
        self.backup = self.db_path + "BACKUP/"
        self.remote_api_base_url = (
            remote_api_base_url.rstrip("/") if remote_api_base_url else None
        )
        self.remote_prefix = remote_prefix.strip("/")

    def set_up_db(self, main_tables=[]):
        if not self.remote_api_base_url:
            if not os.path.isdir(self.db_path):
                os.makedirs(self.db_path, exist_ok=True)
            os.makedirs(self.main, exist_ok=True)
            os.makedirs(self.backup, exist_ok=True)
            os.mkdir(self.backup)
        self.create_table_in_main(main_tables)

    def do_full_backup(self, main_only=True):
        ts = datetime.today().strftime("%Y%m%d-%H%M")
        if not self.remote_api_base_url:
            os.makedirs(self.backup + "FULL", exist_ok=True)
            shutil.copytree(src=self.main, dst=self.backup + f"FULL/MAIN/{ts}")
        else:
            for fname in self._ls("MAIN", ext=".xlsx"):
                # copia cada .xlsx a BACKUP/FULL/MAIN/<ts>/
                src_key = self._remote_key("MAIN", fname)
                dst_key = self._remote_key("BACKUP", "FULL", "MAIN", ts, fname)
                self._upload(
                    dst_key,
                    self._download(src_key),
                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        if not main_only:
            for partition in os.listdir(self.db_path):
                if os.path.isdir(self.db_path + partition) and partition != "BACKUP":
                    shutil.copytree(
                        src=self.db_path + partition,
                        dst=self.backup
                        + f"FULL/{datetime.today().strftime('%Y%m%d-%H%M')}/partition",
                    )

    def create_table_in_main(self, tables):
        if type(tables) == str:
            tables = [tables]
        for table in tables:
            data_empty: dict[str, pd.DataFrame] = {}
            if not self._exists("MAIN", f"{table}.xlsx"):
                self._write_excel("MAIN", f"{table}.xlsx", data_empty)
            if not self._exists("MAIN", f"specs-{table}.xlsx"):
                self._write_excel("MAIN", f"specs-{table}.xlsx", data_empty)

    def write_onto_db(
        self, data, table, partition="MAIN", specs=None, curve_specs=None
    ):
        if type(data) != dict:
            print(
                f"Expected input data as dictionary, got {type(data)} instead; aborting."
            )
            return None
        else:
            # the following should throw error is index is not appropriate
            for instrument in data.keys():
                data[instrument].index = pd.to_datetime(
                    data[instrument].index
                ).normalize()
            if partition == "MAIN" and not os.path.isdir(
                self.db_path + partition + "/" + table
            ):
                raise ValueError(
                    "Automatic table creation in MAIN not allowed: use .create_table_in_main first."
                )
            elif os.path.isdir(self.db_path + partition) and os.path.isdir(
                self.db_path + partition + "/" + table
            ):
                existing_tickers = [
                    ticker
                    for ticker in data.keys()
                    if ticker + ".txt" in self._ls(partition, table)
                ]
                if len(existing_tickers) != 0:
                    if table.split("-")[-1] == "curves":
                        existing_df = {
                            ticker: pd.read_csv(
                                f"{self.db_path}/{partition}/{table}/{ticker}.txt",
                                index_col=0,
                            )
                            for ticker in existing_tickers
                        }
                        existing_specs = {
                            ticker: pd.read_csv(
                                f"{self.db_path}/{partition}/{table}/specs-{ticker}.txt",
                                index_col=0,
                            )
                            for ticker in existing_tickers
                        }
                        existing_curve_specs = pd.read_csv(
                            f"{self.db_path}/{partition}/specs-curves.txt", index_col=0
                        )

                    else:
                        existing_df = {
                            ticker: pd.read_csv(
                                f"{self.db_path}/{partition}/{table}/{ticker}.txt",
                                index_col=0,
                            )
                            for ticker in existing_tickers
                        }
                        if os.path.isfile(
                            f"{self.db_path}/{partition}/{table}/specs-{table}.txt"
                        ):
                            existing_specs = pd.read_csv(
                                f"{self.db_path}/{partition}/{table}/specs-{table}.txt",
                                index_col=0,
                            )
                else:
                    existing_df = {}
                    if table.split("-")[-1] == "curves":
                        existing_specs = {}
                        existing_curve_specs = pd.DataFrame(
                            columns=["Type", "Calendar"]
                        )
                    else:
                        existing_specs = pd.DataFrame(columns=["Placeholder"])
            else:
                existing_df = {}
                if table.split("-")[-1] == "curves":
                    existing_specs = {}
                else:
                    existing_specs = pd.DataFrame(columns=["Placeholder"])

            for instrument in data.keys():
                if instrument in existing_df.keys():
                    temp = data[instrument].merge(
                        existing_df[instrument],
                        left_index=True,
                        right_index=True,
                        suffixes=("", "_old"),
                        how="outer",
                    )

                    for column in [
                        column for column in temp.columns if column[-4:] == "_old"
                    ]:
                        dates_missing_from_new = temp.index[temp[column[:-4]].isna()]
                        temp.loc[dates_missing_from_new, column[:-4]] = temp.loc[
                            dates_missing_from_new, column
                        ]
                    existing_df[instrument] = temp[
                        [column for column in temp.columns if column[-4:] != "_old"]
                    ]

                    if table.split("-")[-1] == "curves":
                        if specs is not None and instrument in specs.keys():
                            temp = existing_specs[instrument].merge(
                                specs[instrument],
                                left_index=True,
                                right_index=True,
                                suffixes=("", "_old"),
                                how="outer",
                            )
                            temp = temp.reindex(
                                temp.index.union(existing_df[instrument].columns)
                            )
                            for column in [
                                column
                                for column in temp.columns
                                if column[-4:] == "_old"
                            ]:
                                tickers_missing_from_new = temp.index[
                                    temp[column[:-4]].isna()
                                ]
                                temp.loc[tickers_missing_from_new, column[:-4]] = (
                                    temp.loc[tickers_missing_from_new, column]
                                )
                            existing_specs[instrument] = temp[
                                [
                                    column
                                    for column in temp.columns
                                    if column[-4:] != "_old"
                                ]
                            ]
                        existing_curve_specs = existing_curve_specs.reindex(
                            existing_curve_specs.index.union([instrument])
                        )
                        if curve_specs is not None and instrument in curve_specs.index:
                            temp = existing_curve_specs.merge(
                                curve_specs,
                                left_index=True,
                                right_index=True,
                                suffixes=("", "_old"),
                                how="outer",
                            )
                            for column in [
                                column
                                for column in temp.columns
                                if column[-4:] == "_old"
                            ]:
                                tickers_missing_from_new = temp.index[
                                    temp[column[:-4]].isna()
                                ]
                                temp.loc[tickers_missing_from_new, column[:-4]] = (
                                    temp.loc[tickers_missing_from_new, column]
                                )
                            existing_curve_specs = temp[
                                [
                                    column
                                    for column in temp.columns
                                    if column[-4:] != "_old"
                                ]
                            ]
                        existing_curve_specs.loc[instrument, "Type"] = table.split("-")[
                            0
                        ]
                else:
                    existing_df[instrument] = data[instrument]
                    if table.split("-")[-1] == "curves":
                        if specs is not None and instrument in specs.keys():
                            existing_specs[instrument] = specs[instrument]
                        else:
                            existing_specs[instrument] = pd.DataFrame(
                                np.nan,
                                index=data[instrument].columns,
                                columns=["Tenor"],
                            )
                    else:
                        existing_specs = existing_specs.reindex(
                            existing_specs.index.union([instrument])
                        )

            timestamp = datetime.today().strftime("%Y%m%d-%H%M")
            backup_folder = self.backup + "WRITINGS/" + timestamp + "/" + table

            for ticker in existing_df.keys():
                src = f"{self.db_path}/{partition}/{table}/{ticker}.txt"
                destination = f"{backup_folder}/{ticker}.txt"
                if os.path.isfile(src):
                    if not os.path.isdir(self.backup + "WRITINGS/"):
                        os.mkdir(self.backup + "WRITINGS/")
                    if not os.path.isdir(self.backup + "WRITINGS/" + timestamp):
                        os.mkdir(self.backup + "WRITINGS/" + timestamp)
                    if not os.path.isdir(
                        self.backup + "WRITINGS/" + timestamp + "/" + table
                    ):
                        os.mkdir(self.backup + "WRITINGS/" + timestamp + "/" + table)
                    shutil.copy(src=src, dst=destination)

                src = f"{self.db_path}/{partition}/{table}/{ticker}-specs.txt"
                destination = f"{self.backup}/WRITINGS/{datetime.today().strftime('%Y%m%d-%H%M')}/{table}/{ticker}-specs.txt"
                if os.path.isfile(src):
                    if not os.path.isdir(destination):
                        os.mkdir(destination)
                    shutil.copy(src=src, dst=destination)

                if not os.path.isdir(self.db_path + partition):
                    os.mkdir(self.db_path + partition)

                if not os.path.isdir(self.db_path + partition + "/" + table):
                    os.mkdir(self.db_path + partition + "/" + table)

                existing_df[ticker].to_csv(
                    f"{self.db_path}/{partition}/{table}/{ticker}.txt"
                )

            if table.split("-")[-1] == "curves":
                for ticker in existing_df.keys():
                    existing_specs[ticker].to_csv(
                        f"{self.db_path}/{partition}/{table}/specs-{ticker}.txt"
                    )
                # if curve_specs is not None:
                #     temp = existing_curve_specs.merge(curve_specs, left_index=True, right_index=True,
                #                                       suffixes=("", "_old"), how="outer")
                #     for column in [column for column in temp.columns if column[-4:] == "_old"]:
                #         tickers_missing_from_new = temp.index[temp[column[:-4]].isna()]
                #         temp.loc[tickers_missing_from_new, column[:-4]] = temp.loc[tickers_missing_from_new, column]
                #     existing_curve_specs = temp[[column for column in temp.columns if column[-4:] != "_old"]]
                # else:
                #     existing_curve_specs = existing_curve_specs.reindex(curve_specs.index.union(existing_df.keys()))
                # existing_curve_specs.loc[data.keys(),"Type"] = table.split("-")[0]
                existing_curve_specs.to_csv(
                    self.db_path + partition + "/" + "/specs-curves.txt"
                )
            else:
                existing_specs.to_csv(
                    f"{self.db_path}/{partition}/{table}/specs-{table}.txt"
                )

    def load_data(
        self, *tickers, table, partition, start_date="19000101", end_date="21000101"
    ):
        existing_tickers = [
            t for t in tickers if f"{t}.txt" in self._ls(partition, table)
        ]

        missing_tickers = [
            ticker for ticker in tickers if ticker not in existing_tickers
        ]
        if len(missing_tickers) != 0:
            raise KeyError(
                "The following tickers do not exist in database", *missing_tickers
            )

        data = {}
        for ticker in existing_tickers:
            data[ticker] = self._read_csv(
                partition, table, f"{ticker}.txt", index_col=0
            )
            data[ticker].index = pd.to_datetime(data[ticker].index).normalize()
            data[ticker] = data[ticker].sort_index()[start_date:end_date]
        return data

    def load_market_instrument(
        self, *tickers, start_date="19000101", end_date="21000101"
    ):
        tables = self._ls_dirs("MAIN")
        curve_tables = [t for t in tables if t.split("-")[-1] == "curves"]
        other_tables = [t for t in tables if t not in curve_tables]

        existing_tickers_dic = {}
        existing_tickers = []
        for table in other_tables:
            temp_existing = [
                t for t in tickers if f"{t}.txt" in self._ls("MAIN", table)
            ]
            existing_tickers_dic[table] = temp_existing
            existing_tickers += temp_existing

        existing_curve_tickers = {}
        for table in curve_tables:
            curves = [
                fname.split(".")[0]
                for fname in self._ls("MAIN", table, ext=".txt")
                if not fname.startswith("specs-")
            ]
            for curve in curves:
                curve_specs = self._read_csv(
                    "MAIN", table, f"specs-{curve}.txt", index_col=0
                )
                for ticker in tickers:
                    if ticker in curve_specs.index:
                        existing_curve_tickers[ticker] = (table, curve)
                        existing_tickers.append(ticker)

        missing_tickers = [
            ticker for ticker in tickers if ticker not in existing_tickers
        ]
        if len(missing_tickers) != 0:
            raise KeyError(
                "The following tickers do not exist in database", *missing_tickers
            )

        data = {}
        for table in existing_tickers_dic.keys():
            for ticker in existing_tickers_dic[table]:
                data[ticker] = self._read_csv(
                    "MAIN", table, f"{ticker}.txt", index_col=0
                )
                data[ticker].index = pd.to_datetime(data[ticker].index).normalize()
                data[ticker] = data[ticker].sort_index()[start_date:end_date]

        for ticker in existing_curve_tickers.keys():
            df = self._read_csv(
                "MAIN",
                existing_curve_tickers[ticker][0],
                f"{existing_curve_tickers[ticker][1]}.txt",
                index_col=0,
            )
            temp = df[[ticker]]
            temp.columns = ["Price"]
            temp.index = pd.to_datetime(temp.index).normalize()
            data[ticker] = temp.sort_index()[start_date:end_date]

        data = {
            ticker: data[ticker].dropna(how="all", axis=0) for ticker in data.keys()
        }
        return data

    def list_market_instruments(
        self,
        table="all",
        include_curve_instruments=False,
        return_result=True,
        print_result=False,
    ):
        if table != "all":
            if table.split("-")[-1] == "curves":
                tickers = []
            else:
                tickers = [
                    folder
                    for folder in self._ls("MAIN", table)
                    if os.path.isdir(f"{self.main}/{table}/{folder}")
                ]
        else:
            tickers = []
            tables = [
                folder
                for folder in os.listdir(self.main)
                if os.path.isdir(self.main + folder)
            ]
            for table in tables:
                tickers += self.list_market_instruments(table=table)
        return tickers

    #     def list_market_instruments(self, include_curve_instruments=False, return_result=True, print_result=False):
    #         specs_files = [file for file in os.listdir(self.main) if file.split(".")[0][:5] == "specs"]
    #         if not include_curve_instruments:
    #             curves_specs = [file for file in specs_files if file.split(".")[0][-6:] == "curves"]
    #             specs_files = [file for file in specs_files if file not in curves_specs]
    #         market_instruments = pd.Index([])
    #         for file in specs_files:
    #             temp_specs = pd.read_excel(self.main + file, index_col=0)
    #             market_instruments = market_instruments.union(temp_specs.index)
    #         if print_result:
    #             print(*market_instruments)
    #         if return_result:
    #             return market_instruments
    #         return market_instruments

    def load_generic_instrument(
        self, *generic_tickers, start_date="19000101", end_date="21000101"
    ):
        # need to filter out invalid generic tickers first, because we override KeyErrors below
        tickers_dic = {
            ticker: compute_generic_asset(
                ticker, start_date=start_date, end_date=end_date
            )
            for ticker in generic_tickers
        }
        data = {ticker: pd.DataFrame(columns="Price") for ticker in generic_tickers}
        for ticker in generic_tickers:
            for sp_ticker in tickers_dic[ticker].index:
                try:
                    temp_data = self.load_market_instrument(
                        sp_ticker,
                        start_date=tickers_dic[ticker].loc[sp_ticker, "start_date"],
                        end_date=tickers_dic[ticker].loc[sp_ticker, "end_date"],
                    )
                except KeyError:
                    continue
                data[ticker] = data[ticker].merge(
                    temp_data,
                    left_index=True,
                    right_index=True,
                    suffixes=("", "_old"),
                    how="outer",
                )
                for column in [
                    column for column in data[ticker].columns if column[-4:] == "_old"
                ]:
                    dates_missing_from_new = data[ticker].index[
                        data[ticker][column[:-4]].isna()
                    ]
                    data[ticker].loc[dates_missing_from_new, column[:-4]] = data[
                        ticker
                    ].loc[dates_missing_from_new, column]
                data[ticker] = data[ticker][
                    [column for column in data[ticker].columns if column[-4:] != "_old"]
                ]
        return data

    def load_curve(self, *tickers, start_date="19000101", end_date="21000101"):
        if start_date is None or end_date is None:
            print("Must provide start and end dates")
            return None
        # start_date = pd.to_datetime(start_date)
        # end_date = pd.to_datetime(end_date)
        data = {}
        curve_specs = self._read_csv("MAIN", "", "specs-curves.txt", index_col=0)
        existing_tickers = [ticker for ticker in tickers if ticker in curve_specs.index]
        missing_tickers = [
            ticker for ticker in tickers if ticker not in curve_specs.index
        ]
        if len(missing_tickers) != 0:
            raise KeyError(
                "The following curves do not exist in database", *missing_tickers
            )

        for ticker in existing_tickers:
            data[ticker] = {}
            temp_df = self._read_csv(
                "MAIN",
                f"/{curve_specs.loc[ticker, 'Type']}-curves",
                f"{ticker}.txt",
                index_col=0,
            )
            temp_df.index = pd.to_datetime(temp_df.index)
            data[ticker]["Data"] = (
                temp_df[start_date:end_date].dropna(how="all", axis=1).sort_index()
            )
            data[ticker]["Specs"] = self._read_csv(
                "MAIN",
                f"/{curve_specs.loc[ticker, 'Type']}-curves",
                f"specs-{ticker}.txt",
                index_col=0,
            )
            data[ticker]["Curve Specs"] = curve_specs.loc[ticker]
        return data

    def _remote_key(self, *parts: str) -> str:
        parts_clean = [
            str(p).strip("/") for p in parts if p is not None and str(p).strip("/")
        ]
        return "/".join([self.remote_prefix] + parts_clean)

    def _ls(
        self,
        partition: str,
        table: str = "",
        ext: str | None = None,
        recursive: bool = False,
    ) -> list[str]:
        if not self.remote_api_base_url:
            base = (
                f"{self.db_path}/{partition}"
                if not table
                else f"{self.db_path}/{partition}/{table}"
            )
            try:
                names = os.listdir(base)
            except FileNotFoundError:
                return []
            files = [n for n in names if os.path.isfile(os.path.join(base, n))]
            return [n for n in files if not ext or n.endswith(ext)]
        # remoto vía API
        url = f"{self.remote_api_base_url}/list"
        params = {"prefix": self._remote_key(partition, table), "recursive": recursive}
        if ext:
            params["ext"] = ext
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        # el backend devuelve rutas relativas al bucket; nos quedamos solo con el nombre de archivo
        return [p.split("/")[-1] for p in r.json().get("files", [])]

    def _read_csv(self, partition: str, table: str, filename: str, **kwargs):
        if not self.remote_api_base_url:
            base = (
                f"{self.db_path}/{partition}"
                if not table
                else f"{self.db_path}/{partition}/{table}"
            )
            return pd.read_csv(f"{base}/{filename}", **kwargs)
        url = f"{self.remote_api_base_url}/download"
        key = self._remote_key(partition, table, filename)
        r = requests.get(url, params={"key": key}, timeout=60)
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), **kwargs)

    def _ls_dirs(self, partition: str, parent: str = "") -> list[str]:
        if not self.remote_api_base_url:
            base = (
                f"{self.db_path}/{partition}"
                if not parent
                else f"{self.db_path}/{partition}/{parent}"
            )
            try:
                return [
                    n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n))
                ]
            except FileNotFoundError:
                return []

        # remoto: derivar dirs a partir de los ficheros bajo el prefijo (sin tocar la API)
        base_key = self._remote_key(partition, parent)
        url = f"{self.remote_api_base_url}/list"
        seen: set[str] = set()
        start_after: str | None = None
        while True:
            params = {"prefix": base_key, "recursive": True, "limit": 2000}
            if start_after:
                params["start_after"] = start_after
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            files: list[str] = r.json().get("files", [])
            if not files:
                break
            prefix = base_key.rstrip("/") + "/"
            for key in files:
                # key es tipo "DB/HIST/MAIN/foo/bar.txt" o "DB/HIST/MAIN/bar.txt"
                if not key.startswith(prefix):
                    continue
                rest = key[len(prefix) :]
                if "/" in rest:
                    seen.add(rest.split("/", 1)[0])
            if len(files) < params["limit"]:
                break
            start_after = files[-1]
        return sorted(seen)


class BeautifulDataAFSStyleXL:
    def __init__(
        self,
        db_path=readDataPath() + "/Inputs/DB/",
        remote_api_base_url: str = "http://10.66.40.13:15555/api/v1/s3",
        remote_prefix: str = "Inputs/DB",
    ):
        if db_path[-1] != "/":
            db_path = db_path + "/"
        self.db_path = db_path
        self.main = self.db_path + "MAIN/"
        self.backup = self.db_path + "BACKUP/"
        self.remote_api_base_url = (
            remote_api_base_url.rstrip("/") if remote_api_base_url else None
        )
        self.remote_prefix = remote_prefix.strip("/")

    def set_up_db(self, main_tables=[]):
        if not os.path.isdir(self.db_path):
            os.mkdir(self.path)
        if not os.path.isdir(self.main):
            os.mkdir(self.main)
        if not os.path.isdir(self.backup):
            os.mkdir(self.backup)
        self.create_table_in_main(main_tables)

    def do_full_backup(self, main_only=True):
        if not os.path.isdir(self.backup + "FULL"):
            os.mkdir(self.backup + "FULL")
        shutil.copytree(
            src=self.main,
            dst=self.backup + f"FULL/MAIN/{datetime.today().strftime('%Y%m%d-%H%M')}",
        )
        if not main_only:
            for partition in os.listdir(self.db_path):
                if os.path.isdir(self.db_path + partition) and partition != "BACKUP":
                    shutil.copytree(
                        src=self.db_path + partition,
                        dst=self.backup
                        + f"FULL/{datetime.today().strftime('%Y%m%d-%H%M')}/partition",
                    )

    def create_table_in_main(self, tables):
        if type(tables) == str:
            tables = [tables]
        for table in tables:
            if not os.path.isfile(self.main + table + ".xlsx"):
                pd.DataFrame().to_excel(self.main + table + ".xlsx")
            if not os.path.isfile(self.main + "specs-" + table + ".xlsx"):
                pd.DataFrame().to_excel(self.main + "specs-" + table + ".xlsx")

    def write_onto_db(self, data, table, partition="MAIN"):
        if type(data) != dict:
            print(
                f"Expected input data as dictionary, got {type(data)} instead; aborting."
            )
            return None
        else:
            # the following should throw error is index is not appropriate
            for instrument in data.keys():
                data[instrument].index = pd.to_datetime(data[instrument].index)
            if (
                not self.remote_api_base_url
                and os.path.isdir(self.db_path + partition)
                and os.path.isfile(self.db_path + partition + "/" + table + ".xlsx")
            ) or (
                self.remote_api_base_url and self._exists(partition, f"{table}.xlsx")
            ):
                if table[-6:] == "curves":
                    xl, _ = self._open_excel(partition, f"specs-{table}.xlsx")
                    specs = {
                        curve: xl.parse(sheet_name=curve, index_col=0)
                        for curve in xl.sheet_names
                    }
                else:
                    specs = self._read_excel(
                        partition, f"specs-{table}.xlsx", index_col=0
                    )
            else:
                if partition == "MAIN":
                    print(
                        "Automatic table creation in MAIN not allowed: use .create_table_in_main first."
                    )
                    return None
                else:
                    existing_df = {}
                    if table[-6:] == "curves":
                        specs = {}
                    else:
                        specs = pd.DataFrame(columns=["Placeholder"])
            # The following lines deal with tables created in main, but not yet filled
            useless = [
                key
                for key in existing_df.keys()
                if key[:4] == "Hoja" or key[:5] == "Sheet"
            ]
            for key in useless:
                existing_df.pop(key)
                if table[-6:] == "curves":
                    specs.pop(key)
                else:
                    if key in specs.index:
                        specs.drop(key)
            for instrument in data.keys():
                if instrument in existing_df.keys():
                    temp = data[instrument].merge(
                        existing_df[instrument],
                        left_index=True,
                        right_index=True,
                        suffixes=("", "_old"),
                        how="outer",
                    )

                    for column in [
                        column for column in temp.columns if column[-4:] == "_old"
                    ]:
                        dates_missing_from_new = temp.index[temp[column[:-4]].isna()]
                        temp.loc[dates_missing_from_new, column[:-4]] = temp.loc[
                            dates_missing_from_new, column
                        ]
                    existing_df[instrument] = temp[
                        [column for column in temp.columns if column[-4:] != "_old"]
                    ]
                    if table[-6:] == "curves":
                        specs[instrument] = specs[instrument].reindex(
                            specs[instrument].index.union(data[instrument].columns)
                        )
                else:
                    existing_df[instrument] = data[instrument]
                    if table[-6:] == "curves":
                        specs[instrument] = pd.DataFrame(
                            np.nan, index=data[instrument].columns, columns=["Tenor"]
                        )
                    else:
                        specs = specs.reindex(specs.index.union([instrument]))

            self._backup_file(partition, f"{table}.xlsx", group="WRITINGS")

            if not os.path.isdir(
                self.backup + "WRITINGS/" + datetime.today().strftime("%Y%m%d-%H%M")
            ):
                os.mkdir(
                    self.backup + "WRITINGS/" + datetime.today().strftime("%Y%m%d-%H%M")
                )

            src = self.db_path + partition + "/" + table + ".xlsx"
            destination = (
                self.backup
                + "WRITINGS/"
                + datetime.today().strftime("%Y%m%d-%H%M")
                + "/"
                + table
                + ".xlsx"
            )
            if os.path.isfile(src):
                if not os.path.isdir(destination):
                    os.mkdir(destination)
                shutil.copy(src=src, dst=destination)

            self._backup_file(partition, f"specs-{table}.xlsx", group="WRITINGS")

            self._write_excel(
                partition, f"{table}.xlsx", {k: v for k, v in existing_df.items()}
            )

            if table[-6:] == "curves":
                self._write_excel(
                    partition, f"specs-{table}.xlsx", {k: v for k, v in specs.items()}
                )
            else:
                self._write_excel(partition, f"specs-{table}.xlsx", {"Sheet1": specs})

            if table[-6:] == "curves":
                curve_specs = self._read_excel(
                    partition, "specs-curves.xlsx", index_col=0
                )
                curve_specs = curve_specs.reindex(
                    curve_specs.index.union(existing_df.keys())
                )
                for ticker in existing_df.keys():
                    curve_specs.loc[ticker, "Type"] = table.split("-")[0]
                self._write_excel(
                    partition, "specs-curves.xlsx", {"Sheet1": curve_specs}
                )

    def load_data(
        self, *tickers, table, partition, start_date="19000101", end_date="21000101"
    ):
        xl, _ = self._open_excel(partition, f"{table}.xlsx")
        existing_tickers = [ticker for ticker in tickers if ticker in xl.sheet_names]

        missing_tickers = [
            ticker for ticker in tickers if ticker not in existing_tickers
        ]
        if len(missing_tickers) != 0:
            raise KeyError(
                "The following tickers do not exist in database", *missing_tickers
            )

        data = {}
        for ticker in existing_tickers:
            data[ticker] = xl.parse(sheet_name=ticker, index_col=0)[start_date:end_date]
        return data

    def load_market_instrument(
        self, *tickers, start_date="19000101", end_date="21000101"
    ):
        specs_files = [
            file
            for file in self._ls("MAIN", ext=".xlsx")
            if file.split(".")[0].startswith("specs")
        ]
        curves_specs = [
            file for file in specs_files if file.split(".")[0][-6:] == "curves"
        ]
        other_specs = [file for file in specs_files if file not in curves_specs]
        existing_tickers_dic = {}
        existing_tickers = []
        for specs_file in other_specs:
            temp_specs = self._read_excel("MAIN", specs_file, index_col=0)
            temp_existing = [ticker for ticker in tickers if ticker in temp_specs.index]
            existing_tickers_dic[specs_file[6:]] = temp_existing
            existing_tickers += temp_existing

        existing_curve_tickers = {}
        for specs_file in curves_specs:
            temp_specs, _ = self._open_excel("MAIN", specs_file)
            for curve in temp_specs.sheet_names:
                curve_specs = temp_specs.parse(sheet_name=curve, index_col=0)
                for ticker in tickers:
                    if ticker in curve_specs.index:
                        existing_curve_tickers[ticker] = (specs_file[6:], curve)
                        existing_tickers.append(ticker)

        missing_tickers = [
            ticker for ticker in tickers if ticker not in existing_tickers
        ]
        if len(missing_tickers) != 0:
            raise KeyError(
                "The following tickers do not exist in database", *missing_tickers
            )

        data = {}
        for table in existing_tickers_dic.keys():
            for ticker in existing_tickers_dic[table]:
                # data[ticker] = pd.read_excel(self.main + table, sheet_name=ticker, index_col=0)[start_date:end_date].dropna(how="all", axis=1)
                data[ticker] = self._read_excel(
                    "MAIN", table, sheet_name=ticker, index_col=0
                )[start_date:end_date]

        for ticker in existing_curve_tickers.keys():
            xl = self._read_excel(
                "MAIN",
                existing_curve_tickers[ticker][0],
                sheet_name=existing_curve_tickers[ticker][1],
                index_col=0,
            )
            temp = xl[[ticker]]
            temp.columns = ["Price"]
            # data[ticker] = temp.dropna(how="all", axis=1)
            data[ticker] = temp[start_date:end_date]

        data = {
            ticker: data[ticker].dropna(how="all", axis=0) for ticker in data.keys()
        }
        return data

    def list_market_instruments(
        self, include_curve_instruments=False, return_result=True, print_result=False
    ):
        specs_files = [
            f
            for f in self._ls("MAIN", ext=".xlsx")
            if f.split(".")[0].startswith("specs")
        ]
        if not include_curve_instruments:
            curves_specs = [
                file for file in specs_files if file.split(".")[0][-6:] == "curves"
            ]
            specs_files = [file for file in specs_files if file not in curves_specs]
        market_instruments = pd.Index([])
        for file in specs_files:
            temp_specs = self._read_excel("MAIN", file, index_col=0)
            market_instruments = market_instruments.union(temp_specs.index)
        if print_result:
            print(*market_instruments)
        if return_result:
            return market_instruments
        return market_instruments

    def load_generic_instrument(
        self, *generic_tickers, start_date="19000101", end_date="21000101"
    ):
        # need to filter out invalid generic tickers first, because we override KeyErrors below
        tickers_dic = {
            ticker: compute_generic_asset(
                ticker, start_date=start_date, end_date=end_date
            )
            for ticker in generic_tickers
        }
        data = {ticker: pd.DataFrame(columns="Price") for ticker in generic_tickers}
        for ticker in generic_tickers:
            for sp_ticker in tickers_dic[ticker].index:
                try:
                    temp_data = self.load_market_instrument(
                        sp_ticker,
                        start_date=tickers_dic[ticker].loc[sp_ticker, "start_date"],
                        end_date=tickers_dic[ticker].loc[sp_ticker, "end_date"],
                    )
                except KeyError:
                    continue
                data[ticker] = data[ticker].merge(
                    temp_data,
                    left_index=True,
                    right_index=True,
                    suffixes=("", "_old"),
                    how="outer",
                )
                for column in [
                    column for column in data[ticker].columns if column[-4:] == "_old"
                ]:
                    dates_missing_from_new = data[ticker].index[
                        data[ticker][column[:-4]].isna()
                    ]
                    data[ticker].loc[dates_missing_from_new, column[:-4]] = data[
                        ticker
                    ].loc[dates_missing_from_new, column]
                data[ticker] = data[ticker][
                    [column for column in data[ticker].columns if column[-4:] != "_old"]
                ]
        return data

    def load_curve(self, *tickers, start_date, end_date):
        if start_date is None or end_date is None:
            print("Must provide start and end dates")
            return None
        data = {}
        curve_specs = self._read_excel("MAIN", "specs-curves.xlsx", index_col=0)
        existing_tickers = [ticker for ticker in tickers if ticker in curve_specs.index]
        missing_tickers = [
            ticker for ticker in tickers if ticker not in curve_specs.index
        ]
        if len(missing_tickers) != 0:
            raise KeyError(
                "The following curves do not exist in database", *missing_tickers
            )

        for ticker in existing_tickers:
            data[ticker] = {}
            temp_df = self._read_excel(
                "MAIN",
                f"{curve_specs.loc[ticker, 'Type']}-curves.xlsx",
                sheet_name=ticker,
                index_col=0,
            )
            data[ticker]["Data"] = temp_df[start_date:end_date].dropna(
                how="all", axis=1
            )
            data[ticker]["Specs"] = self._read_excel(
                "MAIN",
                f"specs-{curve_specs.loc[ticker, 'Type']}-curves.xlsx",
                sheet_name=ticker,
                index_col=0,
            )
            data[ticker]["Curve Specs"] = curve_specs.loc[ticker]
        return data

    def list_curves(self, return_result=True, print_result=False):
        discount_curves = self._read_excel(
            "MAIN", "specs-curves.xlsx", index_col=0
        ).index
        if print_result:
            print(*discount_curves)
        if return_result:
            return discount_curves
        return discount_curves

    def _remote_key(self, *parts: str) -> str:
        parts_clean = [
            str(p).strip("/") for p in parts if p is not None and str(p).strip("/")
        ]
        return "/".join([self.remote_prefix] + parts_clean)

    def _ls(
        self,
        partition: str,
        parent: str = "",
        ext: str | None = None,
        recursive: bool = False,
    ) -> list[str]:
        if not self.remote_api_base_url:
            base = (
                f"{self.db_path}/{partition}"
                if not parent
                else f"{self.db_path}/{partition}/{parent}"
            )
            try:
                names = os.listdir(base)
            except FileNotFoundError:
                return []
            files = [n for n in names if os.path.isfile(os.path.join(base, n))]
            return [n for n in files if not ext or n.endswith(ext)]

        url = f"{self.remote_api_base_url}/list"
        params = {"prefix": self._remote_key(partition, parent), "recursive": recursive}
        if ext:
            params["ext"] = ext
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return [p.split("/")[-1] for p in r.json().get("files", [])]

    def _exists(self, partition: str, filename: str, parent: str = "") -> bool:
        return filename in self._ls(partition, parent)

    def _download(self, key: str) -> bytes:
        url = f"{self.remote_api_base_url}/download"
        r = requests.get(url, params={"key": key, "as_text": False}, timeout=120)
        r.raise_for_status()
        return r.content

    def _upload(
        self, key: str, content: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        url = f"{self.remote_api_base_url}/upload"
        headers = {"Content-Type": content_type}
        r = requests.put(
            url, params={"key": key}, data=content, headers=headers, timeout=120
        )
        r.raise_for_status()

    def _open_excel(
        self, partition: str, filename: str
    ) -> tuple[pd.ExcelFile, io.BytesIO | None]:
        if not self.remote_api_base_url:
            return pd.ExcelFile(f"{self.db_path}{partition}/{filename}"), None
        key = self._remote_key(partition, filename)
        buf = io.BytesIO(self._download(key))
        return pd.ExcelFile(buf), buf  # buf se puede reutilizar por pandas

    def _read_excel(self, partition: str, filename: str, **kwargs) -> pd.DataFrame:
        if not self.remote_api_base_url:
            return pd.read_excel(f"{self.db_path}{partition}/{filename}", **kwargs)
        key = self._remote_key(partition, filename)
        return pd.read_excel(io.BytesIO(self._download(key)), **kwargs)

    def _write_excel(
        self, partition: str, filename: str, sheets: dict[str, pd.DataFrame]
    ) -> None:
        if not self.remote_api_base_url:
            with pd.ExcelWriter(f"{self.db_path}{partition}/{filename}") as writer:
                for sheet, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet)
            return
        # remoto
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            for sheet, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet)
        buf.seek(0)
        key = self._remote_key(partition, filename)
        self._upload(
            key,
            buf.read(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    def _backup_file(
        self, partition: str, filename: str, group: str = "WRITINGS"
    ) -> None:
        ts = datetime.today().strftime("%Y%m%d-%H%M")
        dst_prefix = f"BACKUP/{group}/{ts}"
        if not self.remote_api_base_url:
            src = f"{self.db_path}{partition}/{filename}"
            dst_dir = f"{self.db_path}{dst_prefix}"
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.isfile(src):
                shutil.copy(src, f"{dst_dir}/{filename}")
            return
        # remoto
        src_key = self._remote_key(partition, filename)
        # si no existe, nada que copiar
        if not self._exists(partition, filename):
            return
        content = self._download(src_key)
        dst_key = self._remote_key(dst_prefix, filename)
        self._upload(
            dst_key,
            content,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


class BeautifulDataAFSStyleT2T:
    def __init__(self, db_path=readDataPath() + "/DB/ID/"):
        if db_path[-1] != "/":
            db_path = db_path + "/"
        self.db_path = db_path
        self.main = self.db_path + "MAIN/"
        self.backup = self.db_path + "BACKUP/"

    def load_market_instrument(
        self,
        *tickers,
        date,
        start_hour="00:00",
        end_hour="23:59:59.9",
        data_type="book",
    ):
        """
        tickers MUST BE GIVEN properly -> be careful

        data is saved in /Inputs/DBID/ticker with the name: ticker + - + data_type + - + offers/trades + - + date('%Y%m%d')

        :param tickers:
        :param data_type: str: raw/book
        :param date:
        :param start_hour: only for book. Default -> 00:00
        :param end_hour: only for book. Default -> 23:59:59.9
        :return: data: dict of dicts, with trades and offers in each ticker
        """
        if data_type not in ["book", "raw"]:
            print(
                f"Expected input data_type as book or raw for {date}, got {data_type} instead; aborting."
            )
            return None

        date = pd.to_datetime(date)
        date_str = date.strftime("%Y-%m-%d")

        data = {ticker: {} for ticker in tickers}
        for ticker in tickers:
            path = self.db_path + "/" + ticker
            if not os.path.isdir(path):
                raise ValueError(f"Ticker {ticker} non-existent")
            else:
                for table in ["offers", "trades"]:
                    try:
                        df = pd.read_csv(
                            self.db_path
                            + f"{ticker}/{ticker}-{data_type}-{table}-{date.strftime('%Y%m%d')}.txt",
                            sep="\t",
                            header=0,
                            index_col=0,
                        )
                    except FileNotFoundError:
                        raise KeyError(f"No {table} data for {ticker} on {date_str}")

                    # filter by time
                    if data_type == "book":
                        df.index = pd.to_datetime(df.index)
                        df = df.sort_index()

                        start = pd.to_datetime(
                            f"{date.strftime('%Y-%m-%d')} {start_hour}"
                        )
                        end = pd.to_datetime(f"{date.strftime('%Y-%m-%d')} {end_hour}")

                        df = df[start:end]

                    data[ticker][table] = df
        return data

    def load_generic_instrument(
        self, *generic_tickers, date, start_hour="00:00", end_hour="23:59:59.9"
    ):
        tickers_dic = {
            ticker: compute_generic_asset(ticker, date=date)
            for ticker in generic_tickers
        }
        data = self.load_market_instrument(
            *tickers_dic.values(), date=date, start_hour=start_hour, end_hour=end_hour
        )
        data = {ticker: data[tickers_dic[ticker]] for ticker in generic_tickers}
        return data

    def write_onto_db(self, data, date, data_type="book"):
        """
        Given dict of dicts for a given day writes in a .txt format in Database

        data.keys (tickers) MUST BE GIVEN properly -> be careful, otherwise wrong directories might be created

        data is saved in /Inputs/DBID/ticker with the name: ticker + - + data_type + - + offers/trades + - + date('%Y%m%d')

        If a .txt already exists in our partition, the code joins both data

        IMPORTANT:
        Raw data in MIBGAS WS form MUST have 'cdoferta'/'cdtrans' as index
        Parsed data must have a Timestamp as index

        data -> dict of dicts -> Each product has one dict with "offers" df and "trades" df
        data_type -> str with 'book' or 'raw'
        date -> the day we are considering
        """
        if type(data) != dict:
            print(
                f"Expected input data as dict for {date}, got {type(data)} instead; aborting."
            )
            return None

        if data_type not in ["book", "raw"]:
            print(
                f"Expected input data_type as book or raw for {date}, got {data_type} instead; aborting."
            )
            return None

        date = pd.to_datetime(date)
        date_str = date.strftime("%Y-%m-%d")

        t1 = time.time()

        for ticker in data.keys():
            path = self.db_path + "/" + ticker

            if not os.path.isdir(path):
                os.mkdir(path)

            for table in ["offers", "trades"]:
                # if we already have something in the .txt, we add the extra information -> FUTURE EXTENSION OF THE CODE
                # try:
                #     data_previous = pd.read_csv(self.db_path + f"{ticker}/{ticker}-{data_type}-{table}-{date.strftime('%Y%m%d')}.txt", sep="\t", header=0, index_col=0)
                #
                #     if data_type == 'book':
                #         data_previous.index = pd.to_datetime(data_previous.index)
                #
                #     temp = data[ticker][table].merge(data_previous, left_index=True, right_index=True, suffixes=("", "_old"), how="outer", sort=True, indicator=True)
                #
                #     old_columns = [column for column in temp.columns if column[-4:] == "_old"]
                #     columns = [column[:-4] for column in old_columns]
                #     temp.loc[temp['_merge'] == 'right_only', columns] = temp.loc[temp['_merge'] == 'right_only',old_columns].values
                #
                #     data[ticker][table] = temp[[column for column in temp.columns if (column[-4:] != "_old") * (column[-6:] != "_merge")]]
                #
                # except FileNotFoundError:
                #     pass

                # write the file
                data[ticker][table].to_csv(
                    self.db_path
                    + f"{ticker}/{ticker}-{data_type}-{table}-{date.strftime('%Y%m%d')}.txt",
                    sep="\t",
                    header=True,
                    index=True,
                )

        print(
            ">> Parsed for day {} finished (type is {}): {}min {}s".format(
                date_str,
                data_type,
                math.trunc(round((time.time() - t1) / 60, 1)),
                round(time.time() - t1, 5),
            )
        )
