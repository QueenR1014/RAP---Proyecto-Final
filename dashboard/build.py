"""
Arma dashboard/index.html a partir de dashboard/src.html, incrustando
data/stock_prices.csv y los resultados de results/.

    python dashboard/build.py            -> dashboard/index.html (se abre con doble clic)
    python dashboard/build.py --fragment -> sin <!doctype>/<head> (para publicar como artifact)
"""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def datos_js():
    d = pd.read_csv(RAIZ / "data" / "stock_prices.csv")
    d.columns = [c.strip() for c in d.columns]
    activos = ["AMZN", "GOOG", "JPM"]
    data = {
        "fechas": [s.strip() for s in d["Date"]],
        "activos": activos,
        "precios": d[activos].values.tolist(),
    }
    a = json.loads((RAIZ / "results" / "acciones_a_star.json").read_text(encoding="utf-8"))
    m = json.loads((RAIZ / "results" / "acciones_MCTS.json").read_text(encoding="utf-8"))
    sep = (",", ":")
    return (
        "const DATA=" + json.dumps(data, separators=sep) + ";\n"
        "const GUARDADOS={astar:" + json.dumps(a, separators=sep)
        + ",mcts:" + json.dumps(m, separators=sep) + "};"
    )


def main():
    fragmento = "--fragment" in sys.argv
    src = (RAIZ / "dashboard" / "src.html").read_text(encoding="utf-8")
    html = src.replace("/*__DATA__*/", datos_js())
    if fragmento:
        salida = Path(sys.argv[sys.argv.index("--fragment") + 1]) if len(sys.argv) > sys.argv.index("--fragment") + 1 else RAIZ / "dashboard" / "fragment.html"
    else:
        html = (
            '<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
            "</head>\n<body>\n" + html + "\n</body>\n</html>\n"
        )
        salida = RAIZ / "dashboard" / "index.html"
    salida.write_text(html, encoding="utf-8")
    print(f"{salida} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
