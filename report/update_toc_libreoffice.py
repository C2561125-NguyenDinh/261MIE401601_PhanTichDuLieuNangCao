"""(Tùy chọn) Cập nhật mục lục bằng LibreOffice: python report/update_toc_libreoffice.py in.docx out.docx out.pdf"""
import subprocess, sys, time, uno
from com.sun.star.beans import PropertyValue
src, out_docx, out_pdf = sys.argv[1:4]
p = subprocess.Popen(["soffice", "--headless", "--invisible", "--norestore",
                      '--accept=socket,host=localhost,port=2002;urp;'])
ctx = None
for _ in range(60):
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
        ctx = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
        break
    except Exception:
        time.sleep(1)
smgr = ctx.ServiceManager
desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
def pv(n, v):
    x = PropertyValue(); x.Name = n; x.Value = v; return x
doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(src), "_blank", 0, (pv("Hidden", True),))
for _ in range(2):
    idx = doc.getDocumentIndexes()
    for i in range(idx.getCount()):
        idx.getByIndex(i).update()
    doc.refresh()
doc.storeToURL(uno.systemPathToFileUrl(out_docx), (pv("FilterName", "MS Word 2007 XML"),))
doc.storeToURL(uno.systemPathToFileUrl(out_pdf), (pv("FilterName", "writer_pdf_Export"),))
print("indexes:", doc.getDocumentIndexes().getCount())
doc.close(True)
try:
    desktop.terminate()
except Exception:
    pass
p.wait(timeout=60)
