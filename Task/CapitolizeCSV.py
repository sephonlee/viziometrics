import os, errno, csv
import locale


output = "/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/overview_category_ef_figure_sub_class_1997-2014_top44_cap.csv"
input = "/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/overview_category_ef_figure_sub_class_1997-2014_top44.csv"


with open(output, 'wb') as outcsv:
    writer = csv.writer(outcsv, dialect='excel')
#     header = ['image_id', 'image_location', 'class_name', 'paper_id', 'probability', 'format', 'image_height', 'image_width', 'file_size']
    header = ["category_name","num_paper","avg_eigen_factor","avg_num_page","avg_figures_page","avg_equation_page","avg_table_page","avg_photo_page","avg_visualization_page","avg_scheme_page"]
    writer.writerow(header)
    with open(input ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
        for row in reader:
            print row
            catname = row.pop(0)

            row.insert(0, catname.title())
            print row
            writer.writerow(row)