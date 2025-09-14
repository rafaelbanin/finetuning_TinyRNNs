import PyPDF2
from PyPDF2 import PageObject
# Notice, only work for PyPDF2==2.10.5, not work for PyPDF2==3.0.X
def concatenate_pdfs(pdf_list, output_filename, n_cols=5, n_rows=3):
    pdf_writer = PyPDF2.PdfFileWriter()

    pdf_readers = [PyPDF2.PdfFileReader(pdf) for pdf in pdf_list]

    max_width = max([reader.getPage(0).mediaBox[2] for reader in pdf_readers])
    max_height = max([reader.getPage(0).mediaBox[3] for reader in pdf_readers])

    merged_page = PageObject.createBlankPage(width=n_cols*max_width, height=n_rows*max_height)

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i*n_cols + j
            reader = pdf_readers[idx]
            # Use the first page of the PDF for this example
            page = reader.getPage(0)
            # Calculate scaling factors
            # x_scale = Decimal(str(max_width / page.mediaBox[2]))
            # y_scale = Decimal(str(max_height / page.mediaBox[3]))
            # merged_page.mergeScaledTranslatedPage(page, x_scale, y_scale, j*max_width, (n_rows - 1 - i)*max_height)
            merged_page.mergeTranslatedPage(page, j*max_width, (n_rows - 1 - i)*max_height)

    pdf_writer.addPage(merged_page)

    with open(output_filename, 'wb') as f:
        pdf_writer.write(f)