# import pandas as pd

# root = "/Users/annastuckert/Documents/GitHub/ViT_facemap/ViT-pytorch"
# df = pd.read_csv(f"{root}/predictions.csv")


def predictions_formatting(df):
    # Remove the 'Unnamed: 0' column from predictions
    df = df.drop(columns=["Unnamed: 0"])
    # shift column 'Name' to first position
    first_column = df.pop("image_names")
    # insert column at positon [0]
    df.insert(0, "image_names", first_column)
    return df


# df = df.drop(columns=["Unnamed: 0"])
# first_column = df.pop("image_names")
# df.insert(0, "image_names", first_column)
# print(df)
