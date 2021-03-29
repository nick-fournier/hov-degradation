import pandas as pd
import os

def txt2csv(inpath, outpath):
    flist = pd.Series(os.listdir(inpath))

    # Load headers data
    # meta = pd.read_csv((inpath + flist[flist.str.contains("meta")]).to_string(index=False).strip())
    headers = pd.read_csv(inpath + "5min_headers.csv", index_col=0, header=0)

    # Filter file list
    flist = flist[flist.str.contains("station_5min_")]

    for f in flist:
        # Fix output file name
        fout = f.replace("d07_text_", "").replace("_", "-").replace("-5min-", "_5min_").replace(".txt", ".csv")

        if not os.path.exists(outpath + fout):
            # Load em
            df_data = pd.read_csv(inpath + f)
            # Add header
            df_data.columns = headers.columns
            # Save
            df_data.to_csv(outpath + fout)

        print("Done formatting " + fout)


