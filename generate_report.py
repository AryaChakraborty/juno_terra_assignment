from fpdf import FPDF, HTMLMixin
pdf = FPDF()

width = 210
height = 297

pdf.add_page()

# Title and Time
pdf.set_font('Helvetica', 'B', 24)
pdf.cell(40, 10, f'Juno Terra Report')
pdf.ln()
pdf.set_font('Helvetica', '', 12)
pdf.cell(40, 10, f'January 2020 Report')
pdf.ln()
pdf.set_font('Helvetica', '', 12)
pdf.set_text_color(194,8,8)
pdf.cell(40, 10, f'Data in the x-axis is substracted from minimum value for better visualization')
pdf.ln(50)

# content
# first page
pdf.image("methane.jpeg", 5, 50, width - 10)
pdf.image("ozone.jpeg", 5, 170, width - 10)
# second page
pdf.add_page()
pdf.image("carbonmonoxide.jpeg", 5, 20, width - 10)
pdf.image("nitrogendioxide.jpeg", 5, 140, width - 10)

pdf.output('test.pdf', 'F')