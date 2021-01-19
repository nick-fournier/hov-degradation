import pandas as pd
import os
from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Pt
from docx.shared import Inches


def aggregate_results(path):
    df_meta = pd.read_csv(path + "data/meta_2020-11-16.csv")
    df_pred = pd.read_csv(path + 'predictions_D7_' + dates + '.csv')

    total = df_meta.loc[df_meta.Type == 'HV'].ID.count()
    analyzed = df_pred.iloc[:, 0].count()
    pred_sup = df_pred.loc[df_pred.preds_classification > 0].iloc[:, 0].count()
    pred_unsup = df_pred.loc[df_pred.preds_unsupervised > 0].iloc[:, 0].count()


    res = {'Total HOVs': total,
           'Analyzed HOVs': analyzed,
           'Identified Misconfiguations (unsupervised)': pred_unsup,
           'Identified Misconfigurations (supervised)': pred_sup,
           'Analysis date': start_date + " to " + end_date}

    return res

if __name__ == '__main__':
    path = "../../experiments/district_7/"
    start_date = '2020-12-06'
    end_date = '2020-12-12'
    dates = start_date + '_to_' + end_date

    agg_results = aggregate_results(path)

    doc = Document()

    para = doc.add_paragraph()
    run = para.add_run()
    font = run.font
    font.name = 'Calibri'
    font.size = Pt(18)
    font.bold = True
    for key in agg_results.keys():
        run.add_text( key + ": " + str(agg_results[key]))
        run.add_break()
    run.add_break(WD_BREAK.PAGE)

    for dir in os.listdir(path + 'results/classification_' + dates):
        doc.add_heading('Sensor: ' + dir, level=1)
        para = doc.add_paragraph()
        run = para.add_run()
        run.add_break()
        for img in os.listdir(path + 'results/classification_' + dates + '/' + dir):
            run.add_picture(path + 'results/classification_' + dates + '/' + dir + '/' + img, width=Inches(3.4))
            run.add_break()
        run.add_break(WD_BREAK.PAGE)


    doc.save(path + '/results/HOV Deg Results Draft (2021_01_19).docx')