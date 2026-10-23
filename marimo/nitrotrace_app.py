import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import base64, copy, csv, io, json, math, re, statistics, time, zlib
    from typing import Literal, Optional

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from pydantic import BaseModel, Field



@app.cell(hide_code=True)
def data():
    # Curated published records (Choi 2020, Bragulla 2026), 32 synthetic benchmark records,
    # the protocol rulebook and the source registry, compressed. Provenance: docs/DATA_SOURCES.md.
    _BLOB = (
        "eNrtfVtvG0mW5l8JCJg1iaIoXkTd3PWgkuviKVuqKbm6sVPwEikyJWU7yWRlJiWzeg3U06LnbTEzwLzsvi2wz/uwWGAfe973R/iX7HfOiYiMTCYlSiWXy3Y0"
        "0GWRGcyMjDhxvu9c4sRfNkZBFmZbZ2lwMY/joNfp7QynveFFkA2vg+wyml60/5wl042Dv3DLYTTeONi4ofVGayOP8jhEqy90KxXmKojbilqrt7/8qzruqUG7"
        "o/CrTf0rFUwmyTQK1OgyHL1SjTScpcl4PsqjZNrELdNwlKTjYb6Y0Y1n87M4yi7DMa6Mw2yURjNqiCsvLkN1mcyz0Dwjm89m8ULhMepsfnYWh2MV5KrbUZNn"
        "W5Noqs6TVPW67UFfXar8Mk3mF5dqQFdVcq467Z56rr7pnZ5sq2A6RoNQ5Wkww4cgXmS419lCPT1Sje52e0/97X9fbD1Tx99sf9ZsK+pIMM8vkzRTeJkkzfnX"
        "eIs0zGbJdEwvjQHAN9M8QE8CegOFL/Y6ajY7G06SGPfq050iugW9v4xOhjsFcju5M/ohP1JnyRzdLHd1gu8w/in6k0b55STMo1EQY0zw5CzK8hBX6cn4THOg"
        "pnjy5rE6C3CxrZ7mCo8fReloHgcpvS8GPLqYqgY9QT/2KojnIY/xOEyjK/TnPE0m6AN+OgmDbJ6GEzymyT2LqP9hGmLk0SXlzrS8X0tNEzx1Og5nIf6D7oVX"
        "Ef4dhW31NZ6Rh5NZmAb5nH6PG84wohl/wP/pp1ke0JjQNX5tXJ2E4zaEJTnLwhT9G+pnklRns3AUhRmEh6YOjfjN8RHjM8LTU56Z4QS3wcX5NMKvNniu8TFO"
        "rsN044DmH9dms+IDT4UV2CSa5mh+lcToynDybONg0Glt4Et0J4i5G3mQ5kMIwsYBruDF5e9ub7/b3nvT2pgOMVIxpi6nvnbRRA8K/ziZpyO9Nk/73K9RYJZE"
        "AKlX3ZZKk2v16DjK0+QinNLieMRDdJ4kOUYtVOf43U9z/EVD4TRryXJR6M/bv/4zbqTXC77vtfd76pnKkzyIW+5yackiamkJgUjR5cFAzUiuM7oPzQem4Sqc"
        "BvwWG0aY8T2G/jxKMWkbB3k6D/H20i39LrzItNQVS69tF/LnSoaNutxWxxCJsziYvtrk1Tci2SABbG+8wbiS6pLhw7D++JcNHkOotEGHhjE4C2Majp4MxHw6"
        "m6fRecSdjCb0IV9UhYjGf6rHbxjkyUTmS0sORqOQm44Vmr1ORWREiIdWRFobeiWNh9fUezQZhzEWG32TJ8NRGMc8cvgmG2KBQ14CrKqrcGg6s3FwHsRZeGfR"
        "cUTksfrb/9xWT6JsNM8yalSIDGk88yRX7UKPkiL+aR7EGKyWur6MRpei9qIpL9NC+Zd0IX6YTHETK0L3EJgnWh9Bb+WORi7UE+tIR0epxulTJe99ui+aCOM4"
        "hkY/PnnNH800iPiw6A/1un5G6xVLovK1o7CG324c9Pb32t1BpY1RYsPvAtyk0+330CJ8PUvoy3X0Q5gHQ6OA2h357Ggy0nsBSwM3GoqsdTd+jTA4k7+WhrjP"
        "gl+l6yEcNKHm/mXV/1jJGKtv+UcynOq7gJsdnp7+8PzLJzR9L/nBENnYWfthDBWBrxZ5OAwuhNEYPeASAqcdXngbaie4IEkjTYOfvAKAVe5mLskaxRPdqxsv"
        "7QonOXwoZNpp71gdw38/FC5tb3fe3BeEvixe+5EjRM7X6j+rXnsb/0Wf8d/uDga+cRx81mLV31JHQe8zUARS/OoZyVxzfeGiZTsiBkZ9yzB8+eiStGl0fg5l"
        "yuPhXNaCyBNM0hWNFEuF6IbAYo5LZs1ER9NRPB9DjAAxmDDwr0R3AZMdZUmezHjkMDtgQ9xB1tDSwTx8zWJwjqnEXYlkF6vj9dDCQRAn04sM81Dod0esykLb"
        "IH3X3MADmBUxbWZJJGIlUlCs2eAVNDmYJmlo4VCr1lXDoV2YFVqZIg1NQvnDmKmsuUKsF5rX3DJP1BmIcTCjRV5o6lE0FuXc+H//R739p/+hurgXhq3AahKy"
        "lzRUtQaMAWpMIEhkjgFZw45Z/tE65oz5Fdkc9HJaDEhC9F3Uq3CWE7nHUDqIvq5Vo22ZGYaMBkybKbjX4UUUE2idvOhvbuPOmzEg1fQnXbZ0drbb/R6EtdHf"
        "aw921bPmzTbPRXAVanCeTzBLEESeHNgQl8lYxdEkyuk38pLoK1CSGMCLNBiFeOdwxlzfGYdAvjBjoH4EqXr+7OTJSwVlBlpDps1U/RymScmEeqQe/aFvzKJH"
        "ImmM1mwaCZknKiF3t7xok2aDFd67JP8FiesuEX87MkMeLGKGVrL0Wr677u3vDfbbvXdjE7SsND9SjT/Q1DSJ8/XbfRYg+qLC+VYKg2rwz7G0YzIe85JZiC+j"
        "MxlddT6fyg+ILnaFzotWL4Sv8VV0Qbi/3bwPiyBJSqG68dLQkIkIEK9G6iL1uqfN90ZwlmHWR9pYHQCE9AUQX23Xy3C4byCmSKNrTJEmCSa8EGAi17C5nduQ"
        "fLK0aFv9OpnHY1KA/X//NwUL+yJM23aZfq5kqtmOudFaccySss2ignPIU0VRGAXx0EZM/1bp/x3YMBe32TAhno5pisE0XdkFL4PBUvJVlcwVzLhoyL42V9q/"
        "wl7RC8VRnY/XNEBYs/+GBohVRksGyL0nyJkQR27f/vJft9WPg5dL7gi9WlpKg9pN7oi+9Ub8odPu/Gp3xC3WienQQ1gnv2fGaAf+oRjjsSUMhO5hBhEQeCe+"
        "YLXsS/ED4ec8u+vTw1kKvyeY0vAC5DwZXd3GDKvt1yKF+jfqH9Pjza+ffEnDgsmbbrIvFVCJscthMaoG+p9F+ZycHdT+6dFdHN5P9Y/PgtGri5QhxbEiD2Cg"
        "uB1hK4UMnTkpPmPCjkOyrc5ZiSXTZAK80FOCRUqO7hzItAjTpmJfcjFj+vfw2k7Ip6xIfwvgVT3n06TUGBC7UI2Toz8CUb+Eul2w15mWjQV8I6zsqoNybZFl"
        "QI4MyDNjihhg1S4A9fCjuWajL/gdEA3ILpNrEPPLBYmmEEHR2uxKD1+PwnAs5DELZwGWdAjHE1xW01yYPYxEaGZ6xZLOHydgEccnL1QIUeYpKlxgthsH1fva"
        "V4PHHaEAzFOmGtqDQPgCapLjfbVibOIpYmuORuzKJx4mJHqTB5zJR8t6/Y2PK8jzNDqbQ6EyJ9bWLXg5DC47yinGL8hcwKkIgtZkEOA7U+gl0lxQhl57t3BL"
        "yIdav8TdCO48Jd5zmeez7GBrK5ynWG+zyaidpBdbmv5tfff8iP7f7e3tbHd3dstQRMRuR2FJndKS+qJYUkfurD9WmovuOUDFQ0faiKaBvRMlCwKyFI1eYe7h"
        "EkhpxFvkVhxJ6CtbQK0m5FnIo0kInLsOo4tL1nhYG6QjeIrZlbo3IE8IDRnpQCGpu7vtnYIz051pTog2xYt7wRp6sVntw2NawyDq4EagQfmisOHMHVvQ4KRv"
        "aO40I9YsGAuRIFml8ym/Ir0IxrCt9tQyDBDLUoRkUKb6NdWWvLdQaFDWGAIHp8BfgfuE6k4ki7UPYy9donvCn4tnZeq0t3XaJxpPc5wFi0w9CiB0ueoP0PJR"
        "82aCzUHFZXYtsUbxdj8Mm967ISZglsV7Y9Gt30soQDWyEB76+SyB6NEiejolPsStm/eR+edGJxvtWoR6SpNMKOT4eCQsQO9WYubqT5chuzOoKbExx23DuEp6"
        "e4z7k7bHZyBhKQgRZTcuDJZUuu3GwRTkY/1pZOGvTNSXR8CmfD7jt6DbcHyYUQLTMcoxWUso6/aVwHMmqETRXIMyBbfABfkgHCGZTsUoBMQx+hK2a3IE3bdE"
        "j5gl3GNGv9ZDX4xzORp94A7x2//yzzxXMngqDs9zSAAvPjG0akMFRAerIQKieg1QrVYFSZvKQXgNu6AwIftMvn7yrAgZWFYw1M1KIYOCM9wlYnADFsN/PXA8"
        "WPTh/cQISDgHjw3tUQ2MZItI6wE0dHtflP9jGqqW5bQHaru4wkZAsyLdbAuaJAiMNGM0HDdktrm0hxm6ECyaIKMF8DRajDQs9BidjRGSBqNvO32Gp9EEwNsj"
        "/8+giw89wAswKmyS6HfaOztiZupWbG0q8gCRQNxHso8oS+IATlwaFtyNBqGpwRFeJDJsO7DGPi+6vXH/mIcILDFzVXk9a3jRH4EluOoKRnAYCz1nQoqPE868"
        "0ci0MiwiawxqDl3gxaOjJS0fmvs0QnNWxA9lwZKym+cOKou8iUzKyNxTtgtLq1Z6H1vrypED4+5g4vn2l39h4nhbmG8cjuDWBTNj0YIcYj7eaEHDJwQ0xiS/"
        "guM3eHjs0FA6SzBjq4gmTnCaCES9C6C9cYNvSJ66nm8IL1isTssjZdYuShedpev+GqMQDKE+LGlZ6XPKZtErTDpz+W6no1Xbs+izjUJATGDImLkccCAf/YF6"
        "9KKwYwtyWevSUJM5/jwLWQJISgL4IjQJgM6jTDdwwFHAsugw0vajGo/YibaODdxIpluRpJfX2VWFMSepg2xnlbigsbOizN5KfGRLtGyWkA1Ebgriciwk2iyj"
        "fiRF97T1T1FYV2zAbeHxhaODHoy7SKrfjX610WUSwefVGZL/I+wOHfGq8ard0NrxqR2hleNP67A/zdCCa8x6vslOHaLewspbLCZOoBM6f7UTrVilVW+a+2R0"
        "kT3/4hXBo5isBkr3mTxN2iQYcQxH22CiTghziqFUEeATukSbh2S8cqv9/fb+/v7fqYs0AE++JkcXPYhsev7ALiQJkVP8vFFYYREGgRT2gfPiBHUdJX/s76lv"
        "MShYHRP2l89mE+a4++29HnGEbuc/9cXnhgeSq3xnjz9uZfSkZimOawOsOvG0SCNtqy/CUUDGkX4ECxSPCqJLm/JaURzPM+MGEV8gDaN1gpWjtSX8bImXcXpB"
        "7ktx4dEX4lugaS7MDWgA8cbR3cVPFknEnZojl0DCwEgGVNp3VTW4oEjgkEGHOLOVBVRdsmE3DZt2EZG9MOf8hUdPp9n8HMohklVq7c9HRZZsvc9M1N8qT8M+"
        "/a/saSjLSUMPdg59qt3imMTmavcDRKpRzMIVWHNFGIs73OSkmNSGvKs+Cu33H6bB9CKs81UY8RhyD0m55uI9uNlhodFkFanp3c3/twvvX7e/fwMTcn0do1Km"
        "Wwesx8yImQsavhExo22+WqyyvfLHPj7yYiOVNjpQhzxaGQtzaaFgbhB6s0sujo0n1p02rS/YJMQUZO27JUGJB+EvG7yeaCo7drZFoxT009Bb8QeQrl4KKdKj"
        "60KIN83YjR6mMzcOmItOkygSqbU7vusakcye0OYHz6OsvOj3Ji5gZjR5HY2jTEjuhJ1k1nd2hQz84++/dwbiNGLTy8SliXpYXGhC9V5cEKW1GwQmwetowswA"
        "PjpqZPxrVlupRRTG44wHlv+kTByWu8JdlbpdZnXCq5yd17ETE6FARuCAGP2dMUlhX3QFz5rk9YHWYOpiHGX66Y1eBwRbsM1EeimTBDErzEW80CEb9A6LRO9c"
        "4AdoqkXrCQPnYHB2F3l5wNDrMk38zvAQ5SplZUjJgQVQre+NVq8o/JZCYtzn5CixQ+xMlB7iJe7Woy9oVCgNnjpsIZQ38RAH2DjY3+t1Ws4VmhO+NCSRx41I"
        "hVH+QAwVwA0IqeB36AxorWGY82SEsdtK58ii0DwQEEcfZVF83+XHzadFjM0hgP8gXy6UwYFNuyLKFKqU7aU0i0wjLIJUewCNncWhiOmraXI9PXAYdpV+2LG2"
        "WWMyscY/LNJ/fAJj96SH//ROJATjpDcCrxv01WHaNASRvIipTmtkf6p+KAc0tWHaYpkNJRgaXqvRIqYtMCninrJVBsETzj6pkn78ChRlcykWq+W3veHMpDCN"
        "m7TTD7Ox3TyTcaRFMhuxlsy0Arv+rzrkHVCR6I2Q0nsKVz3GyNFYXxFXbREbI692iE5kZGRBQH8y0yw7oxzilxlFY6HNGV3208rAwg9IjoFERhjpB6ySpos7"
        "DpKrJshevnmI1n91djlfcNqn/glnvqUpDSmFnnjfVRpeyN4qQhn6alkI+NY9nSo7zxyHSLnnL6nvzirrDUtsa0hK0llnp0bRaKHXYQeHnNv1UJqe0io7VKWF"
        "I+Qc8xbqbRaFQWAeUyJ+zhr7HrFeMjiYZ5d1ILsCbu2XQVQecl5p8cLRD3ligy93XBXrT/kh0l4XOgYaafZG3dHdJdGVty9Sne2rAt5o+M9Ck0hDcm4tDPSe"
        "LRkySUnEL/BW05KE4DmP0QZ5CmdRzIMDvDwLtduDzDQ0YswiPhCRYpU3msxHl5S8LmluWRikI87FpVAtXcd3AiE1ItYHEcoImSbD5HzoZBYaIXsSTjhrg/pt"
        "M5M1iBJ4KE1VSp4HSwVKovbCZI+b2Sc3h5UJ0w9xVttHJanOSNLqX5l5rdHuZ/j9K8N8xmhc3JFnapRiD2mYbrkZ1nTBdlcWp0mPKRwAjbCN4IFxMoE1vQo4"
        "2WdpN+nX0A3IO0SnkQf0zqRUlDFSXcHuOSWlpJfHzoxZJsm8cyKJWOx40WNBsEDMtySrMj7SMY1+pel1YU+8pyYSWoyko+zcAOg7UtXXIe+xneDG42KeaHpa"
        "xofI2CKgGBYzyEPIvgqnt3zFpiJwv+14kcM25rdEh5i4u44Ikt3pwkk2pW4uLbrtYeG1A4mi9RlflSjU93OCFvaFE5qnF3jprXICWnGL0iJzXf0Qw+tLsLzN"
        "DB7TcOL6Cgs+hejt5AwZMYiS0w+YiAdXQRTzyzEGEPaavhRzX0evwKOYT0HCDMQV8SPdCxGoQ1AjgYWpuIoJULXPs/CEtpaXKLVHTLiEGBXRkmwqHZRWZsDk"
        "uce9u1KqEjepmCXlcOW08sKlRaPfnlAlzDjfZun1Cwewdf5Kp2eyP5qkALZzks04q864Ms1u8huWXcWYEjUBx/SEJP1HTNvLw5S78iOmDn/fvjbXHZav5pSN"
        "jcl2Ht/SKt2dnpZVNID+WENrMAkdgWMvoX4rSSsVN393G+QKoYscOz3pM14F3kadmG+oZchGlSRIaM6K195iW5f9/ZajykQtr9rBsAjt1y7aL3hl6ai/Xjna"
        "8Nl0lJCTAuAu3CPztb2Du0YNXzThmC0bCDGBG/usmnVp3AalvpXu17KRlRYU2gT5s1NZagknwhy7L4BQWw6qwkZ66cX0HTLOFIOBHurMe1sfYAEfHo/zISma"
        "Yo6LbBMK4Mt2sxRZLemURdWNcYg1FRHxx23Fk3/HFf2dXWYJRj8Sc0WIFRtKllyRpXTC768JFHt3oAwxynNaOWXnjn0LCLgxVcyQtNxIZEb+aAgPO1y44sNC"
        "j7OVDaPZyiqEE3101k0hW2ecAkrKnphqSqFQ4lMZD05p4oCR8zRL0kbWNOwycOJl5RjZrSrgiFNzc6A/uXyOjsVr9MfjShJHRticzWO87CWCZzEF0GRzRKgt"
        "oRpRcUBVCwxRDbzkWDBY+3xYUmJYXDFFABxBsXY32WsLzoGA8oCekMEq3b5WEmu48o6OjVIeubPsX5BfRwcs3dEuxEFvHbSBOxnUZTOsNrc7caXCLvZydpKz"
        "5I+wwqc1vUGGDGVQE24U3kRtHEAUpeIIWQzhlaZEpXzuKphG05rkkMeQJpSJQPCMglU9/ouTTagDgCpQchY7HZDiMZmRtBLnLMqIMJCxijJpzEZvUXQ7vG21"
        "L+UlrUhzPCJ38E9zPJwikDJilMFWGbNL2kUpsEq2qKQeUdiQBjpkNwu4Uhoz99bTYjQo8eNzO941K2rdvl4bclHxs5CLdgRrdcTAxgbzpEiFjDVBsK4f2U0m"
        "U0STL1OEuUQWIWlb8rZhCEpQLfv9SnY67neOjT12WXM0lBUofmzU59KMkjhxxi4nX9QkVbFMkZcFUE6mZc0S3B1qVy4qa9CqiMPcXYonZzR12jOVs2O2JXF7"
        "A9eqO4ASsM5+dwk+lTvbi6Rjsdch473AkvQqrl6iyIzIxkGCFrFDVJzVaF2heCyFgPWKi8bkRgRvITc6DTpUOzk6Y8tsBRjoFWmKLfE1vmK2no2BwvzHZXfC"
        "pIpkUNdeE1gIrDNibPdJjOzkwsk/T8M7ouoRTHraZKE1ujaOCFDxmICxtPH0qYsNT4FgEcmsBIrBa3NTxYhjiQwU13pXsPZ/iHlSmo4Ja8nzwktHlFCGO2qj"
        "6NA0KQmEHlTI0VTHjRELy02+imXMPKjU86ZJdKbMQdwlDWUfQEKGb0Y3NL5NvKoOJayBnXceLStnIsFOdoLONrb2+2PtWDduAK3LZUxAGELznSt4zg2tDApP"
        "r5c+ZiFsFOoozqSagn0nsdRFstiAB6leWvh7Qw6oU0kdG0KoCznoVjWBBh42m+rvIC+JgV75xRgwlTKhIjY88HvYClt6GPVNzeM4pzzU+wSLoEgdFWcNMMHY"
        "yx23SvfjTorLecwpEyS9rlvrMdcg0P2lHAUyFgkVi/QJxNYon61mKO64on9dqFNW0p8T9vTpGxR+Bfui2n0qUBaST3TCtM1SUZtJNiiyeFLOH5lIMO1ssamh"
        "JnOT8rPFFP8U+qCnrrDc6MrR/GRTE3Piza9JD7pr5NbF++sGRl6kAuSGjQtHpn877S6GIv47eXu9KYg4Br+Lic8yo15ot3EMtOMhtHRBcnIcZz6o9lySYzQJ"
        "XVpp+0O7LhFYzyQfs1hphzGVjwOZiHWZANc+15RbFKwUE3QYrgnXJLKxwsT0SbVToo/N+LQ5cy2TspmQnsq1iWKexrUmxBewlBFUDkuEgSnfJx2wml47GOy7"
        "FA9ezhttMcVtVTxRlt5a2DRlLGp6xaSBtyzwb/+M6DHprIB9V8gIBa/OqC4Bbsd5pe/IU/XePTFr8OB1dvK5r3TFAm9THmhjxuiScpiom7xroExhtaJAASB1"
        "KftSKUm0nIFpvGgryXHZjeZMuuG8QmzY6CJljscupuj+yOiNmo7pUZUUwR11KcgIEzfgZODqau12hufRa8AixeNFUJ3F+gPZTIobwOUwoWyPi2LfN6fXMZOy"
        "5I9ck0sxGwcWeanSMoqm5DCGdBgnJGWM1i46Cqi5wCqhQO6TajD50eUpoembTvceuxiuUWJsfl50QdYjLRvtsi/u0BJ7l19TnqdfHKUpyXDgdTp5Z+7gyGp1"
        "nfRCG0ZDXc7O0HA9N5jhQGfZyDQZxceBY8Oo6GF6dzFtx5tRilkRFzVZNI7H8+0v/82Zn7e//Pe7xV90YpIbHy124ggGFStGg1qRSmTnDCK9vXm55IN2+glR"
        "KbS/kyDA2UR1Ut8dUtZ7kezuRE6kWiv7dc1l4/8sSTa1KG+EZ2W/xVGjzG4KtDlJxd2YyViPKpnAjuQ/sfoAKSYwa+Fh7tMmn5pHBTpElRkPPmtZCdtm9gmR"
        "ySgoL6R3Yp1VDA5jYZNUFtFBSXSg+zhuUOsL4Ix8vaXCKkc3aqcBg/Gj7N3kbUk3DkadKPTcTeYV1YdtAlh7dL80DN2atJx8oh2l+rlLaq8u0VikAW8fXptq"
        "V0v3VkV/WjpzJSyrRhLQCbnp/va/FFF3Ah9sXaBsJnBUKarEdO0u/bdaguSkEMt3E+mCcZGTK6lAd+2ErnhuRV3f5TVk4xlRI2uainuYafXyza2o6F1tpAJZ"
        "TF66mWwmAcLsaTjt8s4GdgnoUDPeHWLZUqdt9Y9ts4vgUEEkkpRwppp8ZNZnYWdYNxt3rORzMBab23/YBVJ7ZrDboQo323CLDTp7lDzZ3efkgXESccpyu9vp"
        "721l293B3s4mLm52ezudzddLOdPX19ftKd+zjaVrkqaz2l9ityRSBvSGJKZ/8g3iOshv543KVgcWZhtjDZVKmkcx0vGPopwnnVmipF1ErPknE0pGoO37PaoG"
        "fp6pXbzdPk3haa/Jefy6KALRs2KzApk0Agni/yVcoJq/PKctktjytNGOj5b6eztfTyW+Z+hMTd5ANYEmmi7N7HINEBPl4CnDawNq56hP0kW+82CwvUNT1uvU"
        "T9n2zu4mLm5297v9zubPd5mz5Z/SPA1pfoZamdw1Wz4C08zY+X2kvviP2HnacUXhKzvzVDRXpuBLvrPCbVSjuFdTpqNfmQ5TTodWUUs9gzE4/Zmmp6W+CWAY"
        "Y3ch/vxTkKJI+PO2+g/qK0qlw8ZyVH/6FoutrZ5QKCeZmVwKynCjrHQI2VPMwhFkDPZogqSyGWrBGIV3IpV3/qHsx8TPj7RxTRy+MseHevWe2tX7vc6cwvSi"
        "0ek8o39UF1WEQ8zuAItnd5sneqc60Z3e1ijLRm3bbGmS0ZgnZmXz+02sUwbFzuFzU2Zg9Ryan2Ed0gAvl2DQKekoutHFqj3tdpoOHQrGMvfblbn/Zm6V5jyn"
        "aaKbSrkfCDfuExjXoXXUntvaAjcuPhvq1rXrjaqZAOKiTWQHc2AMYneWaHcKr1P05HQUtTGxV22euEF14vb7W9Ms3ZpeB+f93d7Ns7bUdpTQJkvb+AI6f37G"
        "a3j6an45fz3fCoGgR2l47U7P4fjPwYjtoCR99ZiHlM1MNlKYiglUyRgPKmN8Qnr6lGkYCTrmDOQMDhm2v7+BmYwqnGwXl18kjieX5mJbdzRKttx+fUnRaJhR"
        "kBPZkiuwBnHbwl9DSccE6R6++K5mY2C1iesvkpRpttJMnsuqIwCW9vzZTixt9Ts1V8A9pqNLwM0rfcpCW/1RkIWwxE3Rb7lFPoRBaKeY3VrIen6xTqmkvrNp"
        "XPJHio3iUyl2bgo07BfbrXrd2+vEr3WqQGepfGi/vIWltFfCHcZVW7afljYz2NFllH5czivW+5JXlds5D6uFLOkbnd5fv8eNgxJ42vGmzQ1t4JgARdvY7rev"
        "rV0d6Evw64esY7m8oe23GfLKrq/B/Td9yT7G0q6v5XI0v81LrVUp/8G2d/0W77S0BamukEBlD71NLR+WO2XTxeqbvHnY/U2ll8WgBTFvrXE2XX/FvAv4XQwF"
        "b+iWuDhvduRkJvZhS8qoUwzKpFJxEGdOGQv2pSrbnSpbnWTvLqU/o7ZTzgkAum7YiCJ6esslTZftNFpwIFkns9iCSNq7NuWygS8Yg74jrY9o/5xKITCy69g/"
        "e1YL13cdOJr82aHWaKsQstrOgcnvtLNBX2GhMiLzocCiaA+Pih4VP2BUNNWVa3ZDV8sie2z02PihYSPlmlxl+gQu/Gsr6Jg5sKFoKT5ai4hGRS1DIcvbkJYe"
        "QeY5bpSvAMOalg4cPsfAK7qoPcgSdhJp/jiMxFIRv357u6jCv1RLzKOhR0OPhh4NPRo+NBoeqokFmmL7NjvOzZYN0tpLQFgHXjVgSFUphzaxYTgOFqvAcLml"
        "u6+Mrhb5tVx1oEiXkLbeb+ox8QEwEUVguh4UPSj+vkBRLxBT5desEHerO19DPiHDCVRi8+ELBd9BdRWK6+HU1m8vVneuylsCJZ184UwR32NVUV3PXmrYS5F1"
        "N3QqVFYojFudQZaB3hHEoiVZ0BTOf1yaDTxlYfeXE3KfzccXYa5XvhQSQ6HoQW97u7ez290e7O/u7BSFj5c5qb4VJGcFEdJJgdmK4rL1DR0a9KdSKQt+0SAm"
        "rrYotsUWex6mtq6wZ0eeHXmPgSdHHzc5gl6s8qPnGhSulxUne1VpDQlOFmRJWtXwJFlorspvrX+YyR30WsdZboPOR0afNExv3HCywEdLcapHJlRFrZYC6aP9"
        "hsWW2qHwlBIH+tqcE03n8A349F6QAbNVQ5sGkCZU56fUZK5GwXnrVIRP9fcHzFXhxpnperkmeb6GE+E27c4qimNG9kZ64wy/oTbHSxTOJgU6ewOvsfd1Mzr3"
        "fMbzGe/t8YTGe3t0zSzI4Xt19/QGzhnqnc5H7PK581FJ3qtzP6/OnwTqzSYpVMnC3hT3EGI+iSW8Ks561sekMGFZ351jxqXUrt8pt9y7ye1j78DFqWucP7S9"
        "nXYF6PO+6plRuZF7mFBpc7ytE6psww+RClVOmUuwbNzEkOduYsieVS3b7Z6nRd7N41mRTwzxiSHvPDGkXJVFdgoY2JFykFRX6YYcEZMsqZGqFhhp4qz6ttWW"
        "VmNkfftS3qTrRNDFUpYLK3kXgsdK70LwYPmJuRBO3RBIY4dP22yuEft451GPweoDGe+lt3Z8xONDTkl1MZyLtIqoOimqujCwKY3Zk+uFeWzLB5YoyTJzWM1K"
        "qKRGgHOnpB7hLZSk3HgpdZWrs42VtOLquVLYqWE3nXB/m56WeFriaYmnJZ6WeFry29ASQiqsFANcBxuCUZ6u3C0BA8do82n0LlnhI0Dc7w1LkerCgRRCNqyg"
        "YCx1gYtq5GJ3mbcQFK3rSalt67CWr93q1PrcEnGiZMXRSU7ZaGni2YtnLz4A4cnLbfjn4w8+/nCbF+BLgz4aTK8yF280kt7LvjfnJzkn6KxAyZqW5RNX7Ak8"
        "crSKHEYiMPbhFC3yaOhteQ+H3pb/MPBQ3sEtLa4xUq/R0vlYjtH2qRuodWf7vQpnuU2pk2OlzRFa+ugY3j1wx4w6Oppg5YFly3DMyV+o/CBHbq5A4nIjN9xP"
        "FyqpChwRMG1R7nyUJ+lHnyfXbe8XwNzufjzAbAFg8kzwwczsUM/sQc9jt8duj93elv2du4jb+zgDALpZQXGrf/839eQrnL2GfwewZZ+pz3HSQocaaFdyCSoE"
        "bH9ECwgZWrQ7L5eRFACBoVuBoHLR3XFnjhngAlBypJFUCy4OnMTpGJ1ddbElJuAHAqAOfOpXris42GvvdbrbDmTKx5KinuFc6dwbsR4IPRB6IPRA+FBAKLoW"
        "KHgBffI5QqQ6Rnr8uDjLjEGJjstDtd5rc8Jad2e7PVgJjrgPgWNnBThebt8Ij/qyC5BUIfgGcNxr44CrDw0c8U7r4SOK8HZ2B/vF/3bd+rxLlzxuetz0uOlx"
        "0+Pmu8NN1rs7gpuETS503gMQkROGw2lHufUKroLGpYblui1y0R4Wy05lqdvyUW+2krNe9NIdVDS0jAk5KY2a/oD8rXbC7YLy4OjB0YOjB8f3Ao6l+1Gh3SHg"
        "koqwjJfLwGosQmd/xInSz06evHzM5zGz4tAV0jpIuA3l7OyIDkheOBHOKoQSgGKRvFwv9nkTyKK0ByulW0HWNKwF2WJ/k0UYZVp/5BCrz1b7aKKaHxDK+mxc"
        "D7IeZD/xTbnTJRBiYCUd4GzLZemgp1YScwt807/W2mMZMmkq42A2xAAOi3laAZv1jR3oPJEGVNE0ia9oT/HCVkAt/cCn6npvrTdIPVZ+ohXSS7XRG4dpS8Jw"
        "lcqi728jbtfRZR9tTfSWnqdJODlLg2nozpP+ClgTh0QMoHvzYmLsiA/19dLsvNuN0QNfov4jLlFfJVb3r+j6JE1mM+JdHOR39Q0k9iqUo2vcA3rmU4dN1hWi"
        "361uil7ndJ4K+3d2CLy5gYw6iu82KlrWkSuJKCcRFEOxKUMh4+7pqKejno56OurpqKejno56Ourp6IOfmPQFf4UcGnq7zyoKh8vtNIUoyXW5gHxQI/iNPrHP"
        "5q+o12MI43xqOOEtzLJo6PDKUuks3ZLqNNtFxkuTXsVzS88tPbf03NJzS88tPbf03NJzyxK3lPyi+7s3tYOPPZwlAkS9oSmhsLP2+0kKV8aqhN9CunJzCci7"
        "HEUeTM6ii3kyzwx5rLo9lxvUeD9RGCSPUEwLp4iLVKykqDUtHY76nVwt3jPH4V726HI63dMeXS6cMPCnln/CTFXG83eYpabrDHk+6vnob5Cm5hO96xO9f5BA"
        "nn0gks+4iyqcYphLRatA7RP1iGbJ3OkR5YanyRmGYhXObg/at2R1G7TTKusWTDSKzYkFUm9pZ7PRSeYIS0c1eujz0Pd7csV47PPY51O0f/fHJh8qOnIxDje1"
        "TtF7nnSGzfHJa6XzYBgaCXOwGOZxTmcRLDLFCeBAyoyKIt/LDuUh3qhiX51xOZtAP/00j1balLaBA534Us0zjC5eH+In183e44+jyGOpJkdZ4bc7O67O3/1o"
        "qzz63VAeaj3Ueqj9ne6GIhRC76wTtYRGm7NL6qtW/Es7oQrPK6+tZWDMRqPJ0E6mrLAVEFnX1AFLuqzsCYNSk0Ma6WOcvZ3pkwE8KN4XFGl5eUj0kOghUQzP"
        "HCcIYNFHP+PxtERki0OBOVJGUSop4l3HEZVvXywBJA2XyBedrnsOhbsSJVErIw1xKB0J2U0Q6bar4qOJx9oGqgiUe3z0+Ojx8WHx0agJj5AeIT8phDzCKlaO"
        "yKsJXkjpuhlIHEJVqrHKEzkXtkBSWT5LKIkD7sjTW+Da0IGtFXC5FlbWAeWp2yGaLNWgtk1BeC1RMcbWg6gHUQ+i7xNE+73+TSBaqIsq+O7K74oGHoY9DH+8"
        "WURs+yGSmCmRfPVtAV6CKRQcldWEa/SSzopZFRvd2W3v7HdQcnm7v7s7wP9qkHjOUJ9ybHQVELttXBzW3yutTjkSGkjmLPWQJdrjrcfb94O3JbHFmh2Z/AQf"
        "7vSQ6SHzw84sqmKPPSoWu7MC1vzFpOEgWUO9+V1LGS4rsJNKLddkEpmyPW+Kg2V7ZXwc0jDVmLx5kiMDSZvP9TjrNnFg9gV9XbwCQ6s0QoIULPtgTEcTBWwJ"
        "e7T1aOut23WA1m6w+7iQCWrD0SJDpK/tUHdLX66gC6U2tdzBw977txR31DMsaiWTBlOQzknv4puJuohM2TreQS6gRvYYn+hKFJKOd4U4K/x5/+0m+XUyLGuT"
        "VXC21NA9RKB3skl6RVl1JRXorhNlfqPMbzyifZKbT3onNbjVu33zycDnw3oD0RuInzxSBsOIDqWDaQhVrpHRYguGqYAe6OZwBHPu/qA4n2HIwmAypDviVdIV"
        "mLjUzoHEp7Y7xmw1rcm+o947P/KY6K2892DlWflNzl0R9oDpAdMD5oftUf3BgI0zMbytRF2jPs5mdN7SE1d4IllabnKgmjG4GU7rHK32l+JOLdyt/WGtDlqC"
        "ZPmL5k3X2xx2uvWgXNfSgeUn4SQB3oUzmNlvf/lX/Jf22xSDYDW1B2UPyt716hHZI7JH5AcyYb+3CawctwxiWKnShcbbf/orgJTPW29KKTuCaujrt7/8CzS1"
        "HMTu1K+7t21bA4+9tYG0VwukPQZSKhPjsdRjqS8L7cHUg6kH03uD6XpHLLyILi6xDcZBGV2NiKG0bbEU7wF9hU0llQKw6xYgWgdA+2sDaL8WQPsMoFHFYcwW"
        "O/5J5EwGj6CfCoJKLovHA48HHg/WdXc+lx2LbFi5+pO3QZ6FRWE6mr6fwzS5Z6Lomzdv/j8dZrH/"
    )
    DATA = json.loads(zlib.decompress(base64.b64decode(_BLOB)).decode())
    CASE_KEYS = sorted(k for k in DATA if k.startswith(('cases/', 'synthetic/')))

    return CASE_KEYS, DATA


@app.cell(hide_code=True)
def engine(DATA):
    # NitroTrace engine, generated from the app/ package by scripts/build_marimo_engine.py — edit the package,
    # not this cell. All amounts are nmol of ammonia-equivalent N; missing evidence is unknown, never zero.

    # ======================= schemas =======================
    # Pydantic schemas for NitroTrace experiment records, evidence and audit results.
    # 
    # Every scientifically meaningful value carries its original unit, bounds, bound
    # type and provenance. Missing values are ``None`` and mean *unknown*, never zero.

    Provenance = Literal["reported", "manual", "derived", "assumed", "synthetic", "hypothetical"]
    BoundType = Literal["hard_limit", "detection_limit", "stated_interval", "assumed_range", "point"]
    Species = Literal["NH3", "NH4+", "N"]
    Basis = Literal["amount", "mass", "concentration_molar", "concentration_mass"]
    Category = Literal["gas", "electrolyte", "components"]
    CoverageState = Literal["quantified", "unknown", "not_applicable", "specified_synthetic"]

    class Evidence(BaseModel):
        """Where a value came from and whether a human has confirmed it."""

        source_id: Optional[str] = None  # key into data/protocols/sources.json, e.g. "S2"
        url: Optional[str] = None
        location: Optional[str] = None  # page / table / figure / section
        quote: Optional[str] = None  # short supporting passage or table-cell reference
        provenance: Provenance = "manual"
        confirmed: bool = False
        note: Optional[str] = None

    class Interval(BaseModel):
        """A collection / exposure window in minutes relative to an experiment start."""

        start_min: float = 0.0
        end_min: float

        @property
        def duration_min(self) -> float:
            return self.end_min - self.start_min

    class ProductMeasurement(BaseModel):
        """An ammonia/ammonium amount, as measured or as a blank/control result.

        ``lower``/``upper`` are in ``unit``; a nondetect has ``nondetect=True`` and
        ``upper`` equal to the detection limit (lower defaults to 0).
        """

        species: Optional[Species] = None
        basis: Optional[Basis] = None
        unit: str
        lower: Optional[float] = None
        upper: Optional[float] = None
        bound_type: BoundType = "stated_interval"
        nondetect: bool = False
        volume_mL: Optional[float] = None  # required for concentrations
        dilution_factor: float = 1.0  # measured concentration x factor = original sample concentration
        interval: Optional[Interval] = None
        n_replicates: Optional[int] = None
        evidence: Evidence = Field(default_factory=Evidence)

    class GasFlow(BaseModel):
        value: float
        unit: Literal["mL/min", "L/min", "sccm", "slpm"]
        volume_basis: Literal["actual", "standard"]
        temperature_K: Optional[float] = None  # gas temperature at the flow measurement (actual volume)
        pressure_Pa: Optional[float] = None
        reference_temperature_K: Optional[float] = None  # flow-controller standard conditions
        reference_pressure_Pa: Optional[float] = None
        evidence: Evidence = Field(default_factory=Evidence)

    class GasImpurity(BaseModel):
        species: str  # e.g. "NH3", "NO", "NOx (as NO)", "N2O"
        nitrogen_atoms: int = 1
        unit: Literal["ppm", "ppb", "mole_fraction", "mol_percent"]
        lower: Optional[float] = None
        upper: Optional[float] = None
        bound_type: BoundType = "stated_interval"
        measured_where: Literal["delivered_to_cell", "upstream_of_purifier", "supplier_specification", "unknown"]
        # True only if this bound covers every reactive-nitrogen species of concern (e.g. a total NOx+NH3 bound,
        # or a synthetic record that defines a single impurity). A bound on one species leaves the others unknown.
        covers_all_reactive_nitrogen: bool = False
        evidence: Evidence = Field(default_factory=Evidence)

    class GasSource(BaseModel):
        id: str
        label: str
        impurity: Optional[GasImpurity] = None  # None = impurity not measured (unknown)
        flow: Optional[GasFlow] = None
        total_volume_L: Optional[float] = None  # alternative to flow x time, actual volume at T, P
        total_volume_temperature_K: Optional[float] = None
        total_volume_pressure_Pa: Optional[float] = None
        exposure: Optional[Interval] = None
        eta_upper: Optional[float] = None  # upper bound on fraction reaching the measured product
        eta_lower: float = 0.0
        eta_basis: Literal["measured", "conservative_upper_bound_1", "assumed"] = "conservative_upper_bound_1"
        evidence: Evidence = Field(default_factory=Evidence)

    class ControlMeasurement(BaseModel):
        """A blank, background or component-release measurement."""

        id: str
        label: str
        kind: Literal["system_blank", "electrolyte_blank", "component_release"]
        covers: list[Category]
        measurement: ProductMeasurement
        conditions_match: Literal["matched", "different", "unknown"] = "unknown"
        conditions_note: Optional[str] = None
        include_as_scenario: bool = False  # use a non-matched control as an explicit what-if
        time_scaling: Literal["none", "linear"] = "none"  # linear: scale amount by observation/ control duration
        contains_nafion: bool = False
        evidence: Evidence = Field(default_factory=Evidence)

    class IsotopeEvidence(BaseModel):
        performed: bool = False
        quantitative: bool = False
        time_series: bool = False
        labeled_impurities_quantified: bool = False  # 15NOx / 15NH3 in 15N2 supply
        note: Optional[str] = None
        evidence: Evidence = Field(default_factory=Evidence)

    class ExperimentContext(BaseModel):
        flowing_gas: Optional[bool] = None
        nox_measured_alongside: Optional[bool] = None
        argon_control_same_electrode: Optional[bool] = None
        electrode_contains_nitrogen: Optional[bool] = None
        gde_contains_nafion: Optional[bool] = None
        electrode_area_cm2: Optional[float] = None
        electrolyte: Optional[str] = None
        note: Optional[str] = None

    class Review(BaseModel):
        """Review state of a record. 'internal' means checked only by the people who built it."""

        status: Literal["internal", "independent"] = "internal"
        reviewer: Optional[str] = None  # role or affiliation, e.g. "electrochemistry postdoc, external"
        date: Optional[str] = None
        outcome: Optional[str] = None  # e.g. "agrees with inputs and verdict", "disputes gas flow"
        note: Optional[str] = None

    class ExperimentRecord(BaseModel):
        case_id: str
        title: str
        record_type: Literal["synthetic", "published", "published_scenario", "user"] = "user"
        description: Optional[str] = None
        observed_product: Optional[ProductMeasurement] = None
        gas_sources: list[GasSource] = Field(default_factory=list)
        controls: list[ControlMeasurement] = Field(default_factory=list)
        declared_coverage: dict[Category, CoverageState] = Field(default_factory=dict)
        coverage_model: Optional[Literal["system_blank", "sum_components"]] = None
        isotope: IsotopeEvidence = Field(default_factory=IsotopeEvidence)
        context: ExperimentContext = Field(default_factory=ExperimentContext)
        assumptions: list[str] = Field(default_factory=list)
        sources: list[str] = Field(default_factory=list)  # source ids used by this record
        expected: Optional[dict] = None  # benchmark expectations (synthetic records only)
        review: Optional[Review] = None  # who has checked this record's inputs and expected outcome

    # ---------------------------------------------------------------- results ----

    Status = Literal[
        "contamination_not_excluded",
        "exceeds_quantified_budget",
        "insufficient_information",
        "input_inconsistency",
    ]

    STATUS_LABELS: dict[str, str] = {
        "contamination_not_excluded": "Contamination-only explanation not excluded",
        "exceeds_quantified_budget": "Observation exceeds the quantified budget",
        "insufficient_information": "Insufficient information",
        "input_inconsistency": "Input inconsistency",
    }

    class TrailStep(BaseModel):
        label: str
        formula: Optional[str] = None
        values: dict[str, str] = Field(default_factory=dict)
        result: Optional[str] = None

    class Contribution(BaseModel):
        source_id: str
        label: str
        category: str
        lower_nmol: float
        upper_nmol: float
        included: bool
        role: Literal["valid_bound", "scenario", "excluded_overlap", "excluded_unmatched", "excluded_invalid"]
        reason: Optional[str] = None
        trail: list[TrailStep] = Field(default_factory=list)

    class Issue(BaseModel):
        code: str
        severity: Literal["error", "warning", "info"]
        message: str
        field: Optional[str] = None

    class UnquantifiedSource(BaseModel):
        category: str
        reason: str

    class Recommendation(BaseModel):
        rule_id: str
        title: str
        text: str
        trigger: str
        references: list[dict]

    class AuditResult(BaseModel):
        case_id: str
        status: Status
        status_label: str
        headline: str
        explanation: list[str]
        coverage: dict[str, str]
        budget_complete: bool
        budget_label: str
        observed_nmol: Optional[dict[str, float]] = None
        budget_valid_nmol: dict[str, float]
        budget_with_scenarios_nmol: dict[str, float]
        contributions: list[Contribution]
        unquantified: list[UnquantifiedSource]
        issues: list[Issue]
        feasible_allocation: Optional[dict] = None
        recommendations: list[Recommendation] = Field(default_factory=list)
        observed_trail: list[TrailStep] = Field(default_factory=list)
        assumptions: list[str] = Field(default_factory=list)
        hypothetical: bool = False  # True when any input has provenance "hypothetical"
        engine_version: str

    # ======================= units =======================
    # Explicit, tested unit conversions.
    # 
    # Internal comparison unit: nanomoles of ammonia-equivalent nitrogen (nmol N).
    # NH3, NH4+ and N each carry one nitrogen atom, so amounts are interchangeable,
    # but mass -> amount conversion needs the molar mass of the *declared* species.
    # Only the unit strings listed here are accepted; anything else is rejected.

    R = 8.314462618  # J mol^-1 K^-1
    ATM_PA = 101325.0

    MOLAR_MASS_G_PER_MOL = {"NH3": 17.031, "NH4+": 18.038, "N": 14.007}

    AMOUNT_UNITS_TO_MOL = {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6, "µmol": 1e-6, "nmol": 1e-9, "pmol": 1e-12}
    MASS_UNITS_TO_G = {"g": 1.0, "mg": 1e-3, "ug": 1e-6, "µg": 1e-6, "ng": 1e-9}
    MOLAR_CONC_TO_MOL_PER_L = {"M": 1.0, "mol/L": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9}
    MASS_CONC_TO_G_PER_L = {"g/L": 1.0, "mg/L": 1e-3, "ug/L": 1e-6, "µg/L": 1e-6, "ng/mL": 1e-6, "ug/mL": 1e-3, "µg/mL": 1e-3}

    BASIS_UNITS = {
        "amount": AMOUNT_UNITS_TO_MOL,
        "mass": MASS_UNITS_TO_G,
        "concentration_molar": MOLAR_CONC_TO_MOL_PER_L,
        "concentration_mass": MASS_CONC_TO_G_PER_L,
    }

    # Strings that look like units but are ambiguous for liquid samples.
    AMBIGUOUS_UNITS = {"ppm", "ppb", "ppt", "%", "wt%"}

    GAS_FRACTION = {"mole_fraction": 1.0, "mol_percent": 1e-2, "ppm": 1e-6, "ppb": 1e-9}
    FLOW_TO_L_PER_MIN = {"mL/min": 1e-3, "L/min": 1.0, "sccm": 1e-3, "slpm": 1.0}

    class UnitError(ValueError):
        """Raised for an ambiguous, unsupported or inconsistent unit."""

        def __init__(self, code: str, message: str):
            super().__init__(message)
            self.code = code

    def basis_for_unit(unit: str) -> str | None:
        for basis, table in BASIS_UNITS.items():
            if unit in table:
                return basis
        return None

    def to_nmol_n(value: float, unit: str, species: str | None, basis: str | None,
                  volume_mL: float | None = None, dilution_factor: float = 1.0) -> tuple[float, str]:
        """Convert one product value to nmol N. Returns (value, formula text)."""
        if species is None:
            raise UnitError("missing_species", "Species (NH3, NH4+ or N) must be declared explicitly.")
        if species not in MOLAR_MASS_G_PER_MOL:
            raise UnitError("unknown_species", f"Unsupported species '{species}'.")
        if basis is None:
            raise UnitError("missing_basis", "Reporting basis (amount, mass or concentration) must be declared explicitly.")
        if unit in AMBIGUOUS_UNITS and basis_for_unit(unit) is None:
            raise UnitError("ambiguous_unit", f"Unit '{unit}' is ambiguous for a liquid-phase product; use mol, g or mol/L forms.")
        unit_basis = basis_for_unit(unit)
        if unit_basis is None:
            raise UnitError("unsupported_unit", f"Unit '{unit}' is not supported.")
        if unit_basis != basis:
            raise UnitError("basis_unit_conflict", f"Unit '{unit}' implies basis '{unit_basis}' but basis '{basis}' was declared.")
        if dilution_factor <= 0:
            raise UnitError("bad_dilution", "Dilution factor must be positive.")

        M = MOLAR_MASS_G_PER_MOL[species]
        if basis == "amount":
            mol = value * AMOUNT_UNITS_TO_MOL[unit]
            text = f"{value:g} {unit} {species} = {mol*1e9:.6g} nmol N"
        elif basis == "mass":
            mol = value * MASS_UNITS_TO_G[unit] / M
            text = f"{value:g} {unit} {species} / {M} g/mol = {mol*1e9:.6g} nmol N"
        else:
            if volume_mL is None or volume_mL <= 0:
                raise UnitError("missing_volume", "A concentration needs the sample/electrolyte volume to give an amount.")
            V = volume_mL / 1000.0
            if basis == "concentration_molar":
                mol = value * MOLAR_CONC_TO_MOL_PER_L[unit] * dilution_factor * V
                text = f"{value:g} {unit} x DF {dilution_factor:g} x {volume_mL:g} mL = {mol*1e9:.6g} nmol N"
            else:
                mol = value * MASS_CONC_TO_G_PER_L[unit] * dilution_factor * V / M
                text = (f"{value:g} {unit} {species} x DF {dilution_factor:g} x {volume_mL:g} mL / {M} g/mol"
                        f" = {mol*1e9:.6g} nmol N")
        return mol * 1e9, text

    def gas_moles(flow_value: float, flow_unit: str, volume_basis: str, duration_min: float,
                  T_K: float | None, P_Pa: float | None,
                  ref_T_K: float | None, ref_P_Pa: float | None) -> tuple[float, str]:
        """Moles of gas delivered: n = P Q t / (R T).

        Actual flows use the gas temperature and pressure at the flow measurement.
        Standardized flows (sccm/slpm) use the flow controller's reference
        conditions; the cell temperature must never be applied to them.
        """
        if flow_unit not in FLOW_TO_L_PER_MIN:
            raise UnitError("unsupported_unit", f"Flow unit '{flow_unit}' is not supported.")
        standardized_unit = flow_unit in ("sccm", "slpm")
        if standardized_unit and volume_basis != "standard":
            raise UnitError("flow_basis_conflict", f"'{flow_unit}' is a standardized flow but the volume basis was declared 'actual'.")
        if volume_basis == "standard":
            if ref_T_K is None or ref_P_Pa is None:
                raise UnitError("missing_reference_conditions",
                                "A standardized flow needs the flow controller's reference temperature and pressure.")
            T, P, cond = ref_T_K, ref_P_Pa, "controller reference"
        else:
            if T_K is None or P_Pa is None:
                raise UnitError("missing_gas_conditions", "An actual volumetric flow needs gas temperature and pressure.")
            T, P, cond = T_K, P_Pa, "actual gas"
        if T <= 0 or P <= 0 or duration_min < 0:
            raise UnitError("bad_value", "Temperature, pressure and duration must be positive.")
        V_m3 = flow_value * FLOW_TO_L_PER_MIN[flow_unit] * duration_min / 1000.0
        n = P * V_m3 / (R * T)
        text = (f"n_gas = P·Q·t/(R·T) = {P:g} Pa x {flow_value:g} {flow_unit} x {duration_min:g} min "
                f"/ ({R} x {T:g} K) [{cond} conditions] = {V_m3*1000:.6g} L -> {n:.6g} mol")
        return n, text

    def gas_moles_from_volume(volume_L: float, T_K: float | None, P_Pa: float | None) -> tuple[float, str]:
        if T_K is None or P_Pa is None:
            raise UnitError("missing_gas_conditions", "A total gas volume needs its temperature and pressure.")
        n = P_Pa * volume_L / 1000.0 / (R * T_K)
        return n, f"n_gas = P·V/(R·T) = {P_Pa:g} Pa x {volume_L:g} L / ({R} x {T_K:g} K) = {n:.6g} mol"

    def gas_fraction(value: float, unit: str) -> float:
        if unit not in GAS_FRACTION:
            raise UnitError("unsupported_unit", f"Gas impurity unit '{unit}' is not supported.")
        return value * GAS_FRACTION[unit]

    # ======================= budget =======================
    # Contamination budget arithmetic: observed product, gas-delivered and control contributions.
    # 
    # All values are nmol of ammonia-equivalent nitrogen (nmol N). Bounds are
    # propagated with plain interval arithmetic; no distributions are invented.

    INTERVAL_TOLERANCE = 0.01  # 1 % relative difference in durations counts as a mismatch

    def nt_durations_differ(a: float, b: float) -> bool:
        return abs(a - b) > INTERVAL_TOLERANCE * max(abs(a), abs(b), 1e-12)

    def measurement_bounds(m: ProductMeasurement, label: str, field: str,
                           issues: list[Issue]) -> tuple[float, float, list[TrailStep]] | None:
        """Normalize a product measurement to a (lower, upper) interval in nmol N."""
        trail: list[TrailStep] = []
        if m.nondetect:
            if m.upper is None:
                issues.append(Issue(code="nondetect_without_limit", severity="error", field=field,
                                    message=f"{label}: a nondetect needs its detection/quantification limit as the upper bound."))
                return None
            lo_raw, hi_raw = 0.0, m.upper
        else:
            lo_raw = m.lower if m.lower is not None else m.upper
            hi_raw = m.upper if m.upper is not None else m.lower
            if lo_raw is None:
                issues.append(Issue(code="missing_value", severity="error", field=field,
                                    message=f"{label}: no value was entered."))
                return None
        if lo_raw > hi_raw:
            issues.append(Issue(code="inverted_interval", severity="error", field=field,
                                message=f"{label}: lower bound exceeds upper bound."))
            return None
        try:
            lo, lo_text = to_nmol_n(lo_raw, m.unit, m.species, m.basis, m.volume_mL, m.dilution_factor)
            hi, hi_text = to_nmol_n(hi_raw, m.unit, m.species, m.basis, m.volume_mL, m.dilution_factor)
        except UnitError as e:
            issues.append(Issue(code=e.code, severity="error", field=field, message=f"{label}: {e}"))
            return None
        kind = "nondetect (0 to limit)" if m.nondetect else m.bound_type
        trail.append(TrailStep(label=f"{label} — normalize to nmol N ({kind})",
                               values={"lower": lo_text, "upper": hi_text},
                               result=f"[{lo:.4g}, {hi:.4g}] nmol N"))
        return lo, hi, trail

    def observed_bounds(rec: ExperimentRecord, issues: list[Issue]):
        if rec.observed_product is None:
            return None
        return measurement_bounds(rec.observed_product, "Observed product", "observed_product", issues)

    def observation_duration(rec: ExperimentRecord) -> float | None:
        op = rec.observed_product
        if op is not None and op.interval is not None:
            return op.interval.duration_min
        return None

    def gas_contribution(src: GasSource, rec: ExperimentRecord, issues: list[Issue],
                         assumptions: list[str]) -> tuple[Contribution | None, str | None]:
        """Upper/lower bound on product nitrogen from impurities in one gas stream.

        Returns (None, reason) when the stream cannot be bounded because information
        is missing (unknown, not zero), and (None, None) after an input error.
        """
        field = f"gas_sources.{src.id}"
        imp = src.impurity
        if imp is None or imp.upper is None:
            return None, f"{src.label}: impurity not measured"
        if src.total_volume_L is None and src.flow is None:
            return None, f"{src.label}: gas flow / delivered volume during the measurement not stated"

        trail: list[TrailStep] = []
        obs_dur = observation_duration(rec)
        if src.exposure is not None:
            duration = src.exposure.duration_min
            if obs_dur is not None and nt_durations_differ(duration, obs_dur):
                issues.append(Issue(code="interval_mismatch", severity="error", field=field,
                                    message=(f"{src.label}: gas exposure ({duration:g} min) does not match the product "
                                             f"collection interval ({obs_dur:g} min).")))
                return None, None
        elif obs_dur is not None:
            duration = obs_dur
            assumptions.append(f"{src.label}: gas exposure taken equal to the product collection interval ({obs_dur:g} min).")
        else:
            duration = None
        if src.total_volume_L is None and duration is None:
            return None, f"{src.label}: exposure duration not stated"

        try:
            if src.total_volume_L is not None:
                n_gas, text = gas_moles_from_volume(src.total_volume_L, src.total_volume_temperature_K,
                                                    src.total_volume_pressure_Pa)
            else:
                f = src.flow
                n_gas, text = gas_moles(f.value, f.unit, f.volume_basis, duration, f.temperature_K, f.pressure_Pa,
                                        f.reference_temperature_K, f.reference_pressure_Pa)
            x_hi = gas_fraction(imp.upper, imp.unit)
            x_lo = gas_fraction(imp.lower, imp.unit) if imp.lower is not None else 0.0
        except UnitError as e:
            issues.append(Issue(code=e.code, severity="error", field=field, message=f"{src.label}: {e}"))
            return None, None
        trail.append(TrailStep(label="Moles of gas delivered", formula="n_gas = P·Q·t / (R·T)",
                               values={"calculation": text}, result=f"{n_gas:.6g} mol"))

        eta_hi = src.eta_upper if src.eta_upper is not None else 1.0
        if src.eta_upper is None or src.eta_basis == "conservative_upper_bound_1":
            assumptions.append(f"{src.label}: delivery/conversion/collection fraction bounded above by 1 "
                               "(explicit conservative scenario, not a fitted efficiency).")
        a = imp.nitrogen_atoms
        upper = n_gas * x_hi * a * eta_hi * 1e9
        lower = n_gas * x_lo * a * src.eta_lower * 1e9
        trail.append(TrailStep(
            label=f"Impurity-derived nitrogen ({imp.species})", formula="U_i = n_gas · x_i,upper · a_i · η_i,upper",
            values={"n_gas": f"{n_gas:.6g} mol", "x_upper": f"{x_hi:.4g} ({imp.upper:g} {imp.unit})",
                    "x_lower": f"{x_lo:.4g}", "a_i": str(a), "η_upper": f"{eta_hi:g} ({src.eta_basis})",
                    "η_lower": f"{src.eta_lower:g}"},
            result=f"[{lower:.4g}, {upper:.4g}] nmol N"))

        role, reason = "valid_bound", None
        if imp.measured_where == "supplier_specification":
            role = "excluded_invalid"
            reason = ("Supplier purity specification, not a measured delivered-gas impurity concentration; "
                      "shown for reference only.")
        elif imp.measured_where == "upstream_of_purifier":
            role = "scenario"
            reason = ("Measured upstream of a purifier: not a measurement of what reached the cell. Used only as a "
                      "what-if (conservative only if the purifier and downstream lines add no nitrogen species).")
        elif imp.measured_where == "unknown":
            role = "scenario"
            reason = "Measurement location relative to the cell is unknown; used only as a what-if."
        return Contribution(source_id=src.id, label=src.label, category="gas", lower_nmol=lower, upper_nmol=upper,
                            included=role == "valid_bound", role=role, reason=reason, trail=trail), None

    def control_contribution(c: ControlMeasurement, rec: ExperimentRecord, issues: list[Issue],
                             assumptions: list[str]) -> Contribution | None:
        field = f"controls.{c.id}"
        b = measurement_bounds(c.measurement, c.label, field, issues)
        if b is None:
            return None
        lo, hi, trail = b

        obs_dur = observation_duration(rec)
        ctl_dur = c.measurement.interval.duration_min if c.measurement.interval else None
        if obs_dur is not None and ctl_dur is not None and nt_durations_differ(obs_dur, ctl_dur):
            if c.time_scaling == "linear":
                k = obs_dur / ctl_dur
                lo, hi = lo * k, hi * k
                assumptions.append(f"{c.label}: scaled linearly in time by {obs_dur:g}/{ctl_dur:g} min (assumed constant release rate).")
                trail.append(TrailStep(label="Linear time scaling", formula="amount × t_obs / t_control",
                                       values={"factor": f"{k:.4g}"}, result=f"[{lo:.4g}, {hi:.4g}] nmol N"))
            elif c.conditions_match == "matched":
                issues.append(Issue(code="interval_mismatch", severity="error", field=field,
                                    message=(f"{c.label}: control interval ({ctl_dur:g} min) does not match the product "
                                             f"collection interval ({obs_dur:g} min); declare a time-scaling model or "
                                             "mark the control as measured under different conditions.")))
                return None
        elif c.conditions_match == "matched" and obs_dur is not None and ctl_dur is None:
            issues.append(Issue(code="control_interval_missing", severity="warning", field=field,
                                message=f"{c.label}: control interval not stated; it is assumed to match the observation."))

        op = rec.observed_product
        if (c.conditions_match == "matched" and op is not None and op.volume_mL and c.measurement.volume_mL
                and nt_durations_differ(op.volume_mL, c.measurement.volume_mL)):
            issues.append(Issue(code="volume_mismatch", severity="warning", field=field,
                                message=(f"{c.label}: control volume ({c.measurement.volume_mL:g} mL) differs from the "
                                         f"sample volume ({op.volume_mL:g} mL); the amount, not the concentration, is compared.")))

        if c.conditions_match == "matched":
            role, reason = "valid_bound", None
        else:
            why = c.conditions_note or ("measured under different conditions" if c.conditions_match == "different"
                                        else "matching to the experiment is not established")
            if c.include_as_scenario:
                role, reason = "scenario", f"Not interchangeable with the experiment ({why}); used only as an explicit what-if."
            else:
                role, reason = "excluded_unmatched", f"Not interchangeable with the experiment ({why})."
        return Contribution(source_id=c.id, label=c.label, category="+".join(c.covers), lower_nmol=lo, upper_nmol=hi,
                            included=role == "valid_bound", role=role, reason=reason, trail=trail)

    def feasible_allocation(contribs: list[Contribution], obs_lo: float, obs_hi: float) -> dict | None:
        """One illustrative allocation inside the independent source ranges reaching the observed interval."""
        inc = [c for c in contribs if c.included]
        lo_sum = sum(c.lower_nmol for c in inc)
        hi_sum = sum(c.upper_nmol for c in inc)
        target = max(obs_lo, lo_sum)
        if not inc or target > min(obs_hi, hi_sum):
            return None
        slack = hi_sum - lo_sum
        frac = 0.0 if slack == 0 else (target - lo_sum) / slack
        parts = {c.label: c.lower_nmol + frac * (c.upper_nmol - c.lower_nmol) for c in inc}
        return {"target_nmol": target, "allocation_nmol": parts,
                "label": "Illustrative feasible explanation — not a reconstruction of the actual experiment."}

    # ======================= coverage =======================
    # Source-coverage model: which contamination categories are bounded, and by what.
    # 
    # A matched whole-system blank may already contain gas, electrolyte and
    # component contributions. When it overlaps other valid measurements of the same
    # category, the record must select a coverage model; otherwise the overlapping
    # sources are not aggregated and the budget is reported as partial.

    CATEGORIES = ("gas", "electrolyte", "components")

    UNKNOWN_REASONS = {
        "gas": "No measured upper bound on reactive-nitrogen impurities in the gas delivered to the cell.",
        "electrolyte": "No matched electrolyte or whole-system background measurement.",
        "components": "No matched measurement of release from nitrogen-containing or contaminated components.",
    }

    def nt_cats(c: Contribution) -> set[str]:
        return set(c.category.split("+"))

    def coverage_resolve(rec: ExperimentRecord, contribs: list[Contribution], gas_unmeasured: list[str],
                issues: list[Issue]) -> tuple[dict[str, str], list[UnquantifiedSource]]:
        """Mutates contribution roles for overlap handling; returns coverage map and unquantified list."""
        system_blanks = [c for c in contribs if c.role == "valid_bound" and c.source_id in
                         {ctl.id for ctl in rec.controls if ctl.kind == "system_blank"}]
        others = [c for c in contribs if c.role == "valid_bound" and c not in system_blanks]
        ambiguous: set[str] = set()

        if len(system_blanks) > 1:
            # Several matched whole-system blanks are alternatives, not additive sources.
            keep = max(system_blanks, key=lambda c: c.upper_nmol)
            for c in system_blanks:
                if c is not keep:
                    c.role, c.included = "excluded_overlap", False
                    c.reason = f"Alternative whole-system blank; the largest upper bound ({keep.label}) is used instead of summing."
            issues.append(Issue(code="multiple_system_blanks", severity="warning",
                                message="Several matched system blanks were entered; only the most conservative one is used."))
            system_blanks = [keep]

        for sb in system_blanks:
            overlap = [o for o in others if nt_cats(o) & nt_cats(sb)]
            if not overlap:
                continue
            model = rec.coverage_model
            if model == "system_blank":
                for o in overlap:
                    o.role, o.included = "excluded_overlap", False
                    o.reason = f"Already contained in the selected whole-system blank ({sb.label}); not added again."
            elif model == "sum_components":
                sb.role, sb.included = "excluded_overlap", False
                sb.reason = "Coverage model 'sum_components' selected; the whole-system blank is shown but not added."
            else:
                shared = set().union(*[nt_cats(o) & nt_cats(sb) for o in overlap])
                # Every category the set-aside blank covered is now ambiguous, not only the shared ones.
                ambiguous |= nt_cats(sb) | set().union(*[nt_cats(o) for o in overlap])
                for c in [sb, *overlap]:
                    c.role, c.included = "excluded_overlap", False
                    c.reason = ("Overlaps another measurement of the same source category and no coverage model was "
                                "selected; not aggregated.")
                issues.append(Issue(code="overlap_unresolved", severity="warning", field="coverage_model",
                                    message=(f"{sb.label} overlaps {', '.join(o.label for o in overlap)} for "
                                             f"{', '.join(sorted(shared))}. Select a coverage model to aggregate them.")))

        coverage: dict[str, str] = {}
        unq: list[UnquantifiedSource] = []
        for cat in CATEGORIES:
            valid = [c for c in contribs if c.included and cat in nt_cats(c)]
            scen = [c for c in contribs if c.role == "scenario" and cat in nt_cats(c)]
            declared = rec.declared_coverage.get(cat)
            if cat in ambiguous and not valid:
                coverage[cat] = "ambiguous_overlap"
                unq.append(UnquantifiedSource(category=cat, reason="Overlapping measurements without a selected coverage model."))
            elif valid and cat == "gas" and not nt_gas_species_complete(rec, valid):
                coverage[cat] = "partial_species"
                measured = sorted({g.impurity.species for g in rec.gas_sources if g.impurity and g.id in {c.source_id for c in valid}})
                unq.append(UnquantifiedSource(category=cat, reason=(
                    f"Only {', '.join(measured)} bounded in the delivered gas; other reactive-nitrogen species "
                    "(e.g. NOx, N2O, NH3) are not bounded.")))
            elif valid:
                coverage[cat] = "quantified"
            elif declared == "specified_synthetic":
                if rec.record_type == "synthetic":
                    coverage[cat] = "specified_synthetic"
                else:
                    issues.append(Issue(code="synthetic_coverage_on_real_record", severity="error",
                                        field=f"declared_coverage.{cat}",
                                        message=f"'{cat}' is declared synthetic-specified on a non-synthetic record."))
                    coverage[cat] = "unknown"
            elif declared == "not_applicable":
                coverage[cat] = "not_applicable"
                issues.append(Issue(code="declared_not_applicable", severity="info", field=f"declared_coverage.{cat}",
                                    message=f"'{cat}' declared not applicable by the user; this is an assumption, not a measurement."))
            elif scen:
                coverage[cat] = "scenario_only"
                unq.append(UnquantifiedSource(category=cat, reason="Only non-interchangeable (what-if) measurements are available: "
                                              + "; ".join(c.label for c in scen)))
            else:
                coverage[cat] = "unknown"
                reason = UNKNOWN_REASONS[cat]
                if cat == "gas" and gas_unmeasured:
                    reason += " Details: " + ", ".join(gas_unmeasured) + "."
                unq.append(UnquantifiedSource(category=cat, reason=reason))
        return coverage, unq

    def nt_gas_species_complete(rec: ExperimentRecord, valid: list[Contribution]) -> bool:
        ids = {c.source_id for c in valid}
        if any(c.source_id not in {g.id for g in rec.gas_sources} for c in valid):
            return True  # a whole-system blank that covers gas bounds every species together
        return any(g.impurity is not None and g.impurity.covers_all_reactive_nitrogen
                   for g in rec.gas_sources if g.id in ids)

    def coverage_complete(coverage: dict[str, str]) -> bool:
        return all(v in ("quantified", "not_applicable", "specified_synthetic") for v in coverage.values())

    # ======================= controls =======================
    # Protocol-based next-control suggestions.
    # 
    # Each rule in data/protocols/rules.json has a transparent trigger evaluated
    # here, and quotes the source passage that motivates it. These are suggestions,
    # not an optimized experimental design.

    DATA_DIR = None  # notebook: records come from the embedded DATA

    def rules() -> dict[str, dict]:
        return {r["rule_id"]: r for r in DATA["protocols/rules.json"]}

    def sources() -> dict[str, dict]:
        return DATA["protocols/sources.json"]

    def triggered_rules(rec: ExperimentRecord, coverage: dict[str, str], contribs: list[Contribution],
                        issues: list[Issue]) -> list[tuple[str, str]]:
        """Returns (rule_id, concrete trigger description) pairs."""
        out: list[tuple[str, str]] = []
        synthetic = rec.record_type == "synthetic"
        where = {g.impurity.measured_where for g in rec.gas_sources if g.impurity is not None}

        if coverage.get("gas") in ("unknown", "partial_species", "scenario_only", "ambiguous_overlap"):
            out.append(("R1_gas_unquantified", f"Gas coverage: {coverage['gas']}."))
        if "supplier_specification" in where:
            out.append(("R2_supplier_spec_only", "A gas impurity value is a supplier specification."))
        if where & {"upstream_of_purifier", "unknown"}:
            out.append(("R3_upstream_of_purifier", "A gas impurity was measured upstream of the purifier or at an unknown point."))
        if coverage.get("electrolyte") not in ("quantified", "not_applicable", "specified_synthetic"):
            out.append(("R4_background_unresolved", f"Electrolyte/system background coverage: {coverage.get('electrolyte')}."))
        if (coverage.get("components") not in ("quantified", "not_applicable", "specified_synthetic")
                or rec.context.electrode_contains_nitrogen):
            why = f"Component coverage: {coverage.get('components')}."
            if rec.context.electrode_contains_nitrogen:
                why += " The electrode/catalyst contains nitrogen."
            out.append(("R5_component_unresolved", why))
        if rec.context.gde_contains_nafion or any(c.contains_nafion for c in rec.controls):
            out.append(("R6_nafion_gde", "A Nafion-containing electrode or component is part of the setup."))
        iso = rec.isotope
        if not synthetic and (not iso.performed or not iso.quantitative or not iso.time_series):
            state = ("not performed" if not iso.performed else "qualitative only" if not iso.quantitative
                     else "single condition only")
            out.append(("R7_isotope_incomplete", f"Isotope evidence: {state}."))
        if iso.performed and not iso.labeled_impurities_quantified:
            out.append(("R8_labeled_impurities", "15N2 experiment recorded without quantified labeled impurities."))
        mismatch = [i for i in issues if i.code in ("interval_mismatch", "volume_mismatch")]
        unmatched = [c for c in contribs if c.role in ("excluded_unmatched", "scenario") and c.category != "gas"]
        if mismatch or unmatched:
            names = [i.message.split(":")[0] for i in mismatch] + [c.label for c in unmatched]
            out.append(("R9_condition_mismatch", "Non-aligned measurements: " + "; ".join(dict.fromkeys(names)) + "."))
        if rec.context.flowing_gas:
            out.append(("R10_fixed_gas_volume", "Continuous gas flow during the measurement."))
        if not synthetic and rec.context.nox_measured_alongside is not True:
            out.append(("R11_nox_alongside", "NOx alongside ammonia: "
                        + ("not reported." if rec.context.nox_measured_alongside is None else "not measured.")))
        op = rec.observed_product
        if not synthetic and op is not None and (op.n_replicates is None or op.n_replicates < 3):
            out.append(("R12_replicates", f"Independent replicates: {op.n_replicates if op.n_replicates else 'not stated'}."))
        return out

    def recommend(rec: ExperimentRecord, coverage: dict[str, str], contribs: list[Contribution],
                  issues: list[Issue]) -> list[Recommendation]:
        rs, src = rules(), sources()
        recs = []
        for rule_id, trig in triggered_rules(rec, coverage, contribs, issues):
            r = rs[rule_id]
            refs = [dict(ref, citation=src.get(ref["source_id"], {}).get("citation"),
                         url=src.get(ref["source_id"], {}).get("url")) for ref in r["references"]]
            recs.append(Recommendation(rule_id=rule_id, title=r["title"], text=r["text"], trigger=trig, references=refs))
        return recs

    # ======================= audit =======================
    # Run a full evidence audit on one experiment record.
    # 
    # Status logic (Section 7 of the build plan):
    #   1. Any input error                                  -> input_inconsistency
    #   2. No observed product                              -> insufficient_information
    #   3. Valid budget upper >= observed lower             -> contamination_not_excluded
    #      (unknown sources can only raise the budget, so this holds regardless of coverage)
    #   4. Observation above the valid budget, and any
    #      category unknown / ambiguous / scenario-only     -> insufficient_information
    #   5. Otherwise                                        -> exceeds_quantified_budget

    ENGINE_VERSION = "0.3.0"

    def nt_bounds(contribs, roles):
        sel = [c for c in contribs if c.role in roles]
        return {"lower": sum(c.lower_nmol for c in sel), "upper": sum(c.upper_nmol for c in sel)}

    def run_audit(rec: ExperimentRecord) -> AuditResult:
        issues: list[Issue] = []
        assumptions: list[str] = list(rec.assumptions)

        obs = observed_bounds(rec, issues)
        contribs, gas_unmeasured = [], []
        for g in rec.gas_sources:
            c, unknown_reason = gas_contribution(g, rec, issues, assumptions)
            if c is not None:
                contribs.append(c)
            elif unknown_reason:
                gas_unmeasured.append(unknown_reason)
        for ctl in rec.controls:
            c = control_contribution(ctl, rec, issues, assumptions)
            if c is not None:
                contribs.append(c)

        cov, unq = coverage_resolve(rec, contribs, gas_unmeasured, issues)
        is_complete = coverage_complete(cov)
        valid = nt_bounds(contribs, {"valid_bound"})
        with_scen = nt_bounds(contribs, {"valid_bound", "scenario"})
        errors = [i for i in issues if i.severity == "error"]

        explanation: list[str] = []
        observed = None
        allocation = None
        if obs is not None:
            observed = {"lower": obs[0], "upper": obs[1]}

        if errors:
            status = "input_inconsistency"
            headline = "The inputs cannot be compared on a single valid basis; fix the listed issues first."
            explanation += [f"{e.message}" for e in errors]
            explanation.append("This concerns the entered record, not the validity of the underlying study.")
        elif observed is None:
            status = "insufficient_information"
            headline = "No observed product amount was entered, so no comparison is possible."
            explanation.append("The contamination budget below is shown for reference only.")
        elif valid["upper"] >= observed["lower"]:
            status = "contamination_not_excluded"
            headline = ("Under these assumptions, the measured ammonia can be accounted for by the allowed contamination "
                        "budget. This does not demonstrate that contamination actually caused it.")
            explanation.append(f"Valid contamination upper bound {valid['upper']:.4g} nmol N ≥ observed lower bound "
                               f"{observed['lower']:.4g} nmol N.")
            if not is_complete:
                explanation.append("Coverage is partial, but unquantified sources could only enlarge the budget, so this "
                                   "conclusion does not depend on them.")
            allocation = feasible_allocation(contribs, observed["lower"], observed["upper"])
        elif not is_complete:
            status = "insufficient_information"
            missing = [k for k, v in cov.items() if v not in ("quantified", "not_applicable", "specified_synthetic")]
            headline = ("The observation is above the quantified part of the budget, but important sources are not "
                        f"bounded ({', '.join(missing)}); no conclusion about the nitrogen source follows.")
            explanation.append(f"Observed lower bound {observed['lower']:.4g} nmol N > valid partial budget upper bound "
                               f"{valid['upper']:.4g} nmol N.")
            explanation.append("A gap above quantified contamination does not establish that all other sources have been excluded.")
            if with_scen["upper"] != valid["upper"]:
                rel = "≥" if with_scen["upper"] >= observed["lower"] else "<"
                explanation.append(f"Including the what-if (non-interchangeable) measurements gives an upper bound of "
                                   f"{with_scen['upper']:.4g} nmol N ({rel} observed lower bound); this is not a valid bound.")
        else:
            status = "exceeds_quantified_budget"
            headline = ("The measured ammonia exceeds the quantified contamination budget. Attribution still depends on "
                        "the other relevant sources and controls; this does not prove nitrogen reduction.")
            explanation.append(f"Observed lower bound {observed['lower']:.4g} nmol N > budget upper bound "
                               f"{valid['upper']:.4g} nmol N (margin ×{observed['lower']/valid['upper']:.3g})."
                               if valid["upper"] > 0 else
                               f"Observed lower bound {observed['lower']:.4g} nmol N > budget upper bound 0 nmol N.")
            if any(v == "specified_synthetic" for v in cov.values()):
                explanation.append("Some categories are fully specified only because this is a synthetic record.")

        evidence = [rec.observed_product.evidence] if rec.observed_product else []
        evidence += [g.impurity.evidence for g in rec.gas_sources if g.impurity] + [g.evidence for g in rec.gas_sources]
        evidence += [c.measurement.evidence for c in rec.controls] + [c.evidence for c in rec.controls]
        hypothetical = any(e.provenance == "hypothetical" for e in evidence)
        if hypothetical:
            explanation.append("This result includes hypothetical measurements. It shows what they would imply; it is "
                               "not based on evidence that exists.")

        if rec.isotope.performed:
            explanation.append("Isotope evidence is recorded as a control check and does not change the contamination "
                               "budget; an isotope signal alone does not exclude labeled impurities [S2].")

        recs = recommend(rec, cov, contribs, issues)
        if is_complete:
            label = "Budget covers all modeled source categories (under stated assumptions)"
        else:
            label = "Partial budget — incomplete source coverage"

        return AuditResult(
            case_id=rec.case_id, status=status, status_label=STATUS_LABELS[status], headline=headline,
            explanation=explanation, coverage=cov, budget_complete=is_complete, budget_label=label,
            observed_nmol=observed, budget_valid_nmol=valid, budget_with_scenarios_nmol=with_scen,
            contributions=contribs, unquantified=unq, issues=issues, feasible_allocation=allocation,
            recommendations=recs, observed_trail=obs[2] if obs else [],
            assumptions=list(dict.fromkeys(assumptions)), hypothetical=hypothetical, engine_version=ENGINE_VERSION,
        )

    # ======================= reports =======================
    # JSON, CSV and Markdown exports that reproduce the displayed audit.

    DISCLAIMER = ("NitroTrace is a software prototype. Results describe what the entered measurements and stated "
                  "assumptions permit; they are not a laboratory validation or a judgement of the original study.")

    def to_json(rec: ExperimentRecord, res: AuditResult) -> str:
        return json.dumps({"record": rec.model_dump(mode="json"), "audit": res.model_dump(mode="json"),
                           "disclaimer": DISCLAIMER}, indent=2, ensure_ascii=False)

    def to_csv(rec: ExperimentRecord, res: AuditResult) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["case_id", "row_type", "source_id", "label", "category", "role", "included",
                    "lower_nmol_N", "upper_nmol_N", "note"])
        if res.observed_nmol:
            w.writerow([rec.case_id, "observed", "observed_product", "Observed product", "", "", "",
                        f"{res.observed_nmol['lower']:.6g}", f"{res.observed_nmol['upper']:.6g}", ""])
        for c in res.contributions:
            w.writerow([rec.case_id, "contribution", c.source_id, c.label, c.category, c.role, c.included,
                        f"{c.lower_nmol:.6g}", f"{c.upper_nmol:.6g}", c.reason or ""])
        for u in res.unquantified:
            w.writerow([rec.case_id, "unquantified", "", "", u.category, "unknown", False, "", "", u.reason])
        w.writerow([rec.case_id, "budget_valid", "", res.budget_label, "", "", "",
                    f"{res.budget_valid_nmol['lower']:.6g}", f"{res.budget_valid_nmol['upper']:.6g}", ""])
        w.writerow([rec.case_id, "status", "", res.status_label, "", "", "", "", "", res.headline])
        return buf.getvalue()

    def nt_fmt(x: float) -> str:
        return f"{x:.4g}"

    NOT_SHOWN = [
        "Whether nitrogen reduction occurred. An 'exceeds' verdict means only that the measured ammonia is larger than "
        "the contamination that was bounded; isotope-labelled and other controls are still required.",
        "Whether the original study is right or wrong. The audit describes what the entered evidence permits.",
        "Anything about sources that were never measured, other than that they remain unbounded.",
    ]

    def nt_review_line(rec: ExperimentRecord) -> str:
        r = rec.review
        if r is None or r.status == "internal":
            return ("**Internal review only.** The inputs and expected outcome have been checked by the NitroTrace "
                    "authors against the cited source, but not yet by an independent domain researcher. "
                    "Use `nitrotrace review-pack` to produce a checklist for one.")
        who = r.reviewer or "an independent reviewer"
        return f"**Independently reviewed** by {who}{f' on {r.date}' if r.date else ''}: {r.outcome or 'no outcome recorded'}."

    def nt_value(m) -> str:
        sp = f" {m.species}" if getattr(m, "species", None) else ""
        if getattr(m, "nondetect", False):
            return f"not detected (< {m.upper:g} {m.unit}{sp})"
        lo, hi = m.lower, m.upper
        if lo is not None and hi is not None and lo != hi:
            return f"{lo:g}–{hi:g} {m.unit}{sp}"
        v = hi if hi is not None else lo
        return f"{v:g} {m.unit}{sp}" if v is not None else "not stated"

    def nt_ev_row(name: str, value: str, ev) -> str:
        q = (ev.quote or "").replace("|", "/")
        loc = f"[{ev.source_id}] {ev.location or ''}".strip() if ev.source_id else (ev.location or "—")
        return (f"| {name} | {value} | {loc} | {('“' + q + '”') if q else '—'} | {ev.provenance}"
                f"{' ✓' if ev.confirmed else ''} |")

    def to_casefile(rec: ExperimentRecord, res: AuditResult, req: dict | None) -> str:
        """A complete, source-linked account of one case: reported values, what was counted or set aside,
        the calculation, the qualified conclusion, what would close the gap, and the review status."""
        src = sources()
        L = [f"# Case file — {rec.title}", "",
             f"> **Question.** Given this ammonia measurement and these controls, what contamination remains unbounded, "
             f"and how stringent would additional measurements have to be to close that gap?", "",
             f"- **Case:** `{rec.case_id}` ({rec.record_type})" + (" — **contains hypothetical inputs**" if res.hypothetical else ""),
             f"- **Verdict:** {res.status_label}",
             f"- **Review status:** {nt_review_line(rec)}",
             f"- **Engine:** v{res.engine_version}", ""]
        if rec.description:
            L += [rec.description, ""]
        L += ["## 1. Sources", ""]
        L += [f"- [{s}] {src[s]['citation']} {src[s].get('open_full_text') or src[s].get('url', '')}" for s in rec.sources if s in src] or ["- None (synthetic record)."]
        L += ["", "## 2. Measurements as reported", "",
              "| Quantity | Value as reported | Where | Supporting quote | Provenance |", "|---|---|---|---|---|"]
        if rec.observed_product:
            L.append(nt_ev_row("Observed ammonia", nt_value(rec.observed_product), rec.observed_product.evidence))
        for g in rec.gas_sources:
            if g.impurity:
                L.append(nt_ev_row(f"{g.label} — {g.impurity.species} ({g.impurity.measured_where.replace('_', ' ')})",
                                 nt_value(g.impurity), g.impurity.evidence))
            else:
                L.append(nt_ev_row(f"{g.label} — impurity", "not measured", g.evidence))
            if g.flow:
                L.append(nt_ev_row(f"{g.label} — flow", f"{g.flow.value:g} {g.flow.unit} ({g.flow.volume_basis})", g.flow.evidence))
        for c in rec.controls:
            L.append(nt_ev_row(f"{c.label} ({c.kind.replace('_', ' ')}, {c.conditions_match})", nt_value(c.measurement),
                             c.measurement.evidence))
        L += ["", "## 3. Calculation (all amounts in nmol of ammonia-equivalent N)", ""]
        for step in res.observed_trail + [s for c in res.contributions for s in c.trail]:
            L.append(f"- **{step.label}**" + (f" — `{step.formula}`" if step.formula else ""))
            L += [f"  - {k}: {v}" for k, v in step.values.items()]
            if step.result:
                L.append(f"  - → {step.result}")
        L += ["", "## 4. Counted and set aside", "", "| Source | Role | Lower | Upper | Why |", "|---|---|---:|---:|---|"]
        for c in res.contributions:
            L.append(f"| {c.label} | {c.role} | {nt_fmt(c.lower_nmol)} | {nt_fmt(c.upper_nmol)} | {c.reason or 'matched, valid bound'} |")
        if not res.contributions:
            L.append("| — | — | | | No source could be bounded. |")
        L += ["", f"**Valid contamination budget:** [{nt_fmt(res.budget_valid_nmol['lower'])}, "
                   f"{nt_fmt(res.budget_valid_nmol['upper'])}] nmol N ({res.budget_label})."]
        if res.observed_nmol:
            L.append(f"**Observed:** [{nt_fmt(res.observed_nmol['lower'])}, {nt_fmt(res.observed_nmol['upper'])}] nmol N.")
        L += ["", "## 5. Still unbounded (never treated as zero)", ""]
        L += [f"- **{u.category}:** {u.reason}" for u in res.unquantified] or ["- None."]
        L += ["", "## 6. Assumptions", ""] + ([f"- {a}" for a in res.assumptions] or ["- None."])
        L += ["", "## 7. Qualified conclusion", "", f"**{res.status_label}.** {res.headline}", ""]
        L += [f"- {e}" for e in res.explanation]
        if req:
            L += ["", "## 8. What would close the gap", "",
                  f"- Headroom for unbounded sources: **{req['headroom_nmol']:+.4g} nmol N** "
                  f"(observed lower bound − valid budget upper bound).",
                  f"- Contamination floor of this setup: **{req['floor_nmol']:.4g} nmol N** — the smallest observation "
                  "today's controls could call 'above budget'."]
            for u in req["unknown"]:
                for s in u["specs"]:
                    L.append(f"- **{u['category']}:** {s['text']}")
                    L += [f"  - Assumes: {a}" for a in s.get("assumptions", [])]
            for s in req["shrink"]:
                L.append(f"- **Tighten {s['source']}:** {s['text']}")
                L += [f"  - Assumes: {a}" for a in s.get("assumptions", [])]
            L += [f"- {n}" for n in req["notes"]]
            L += [f"- _{c}_" for c in req["caveats"]]
        L += ["", "## 9. What this case file does not show", ""] + [f"- {x}" for x in NOT_SHOWN]
        L += ["", "---", "", f"_{DISCLAIMER}_", ""]
        return "\n".join(L)

    def to_review_pack(rec: ExperimentRecord, res: AuditResult, req: dict | None) -> str:
        """A checklist for an independent domain researcher to confirm or dispute each input and the outcome."""
        L = [f"# Independent review — {rec.title}", "",
             "Thank you for reviewing. Please check each item against the cited source and your own judgement, tick one "
             "box, and add a note where you disagree. Return this file; the outcome is recorded in the record's "
             "`review` field and shown in every report.", "",
             f"- Case: `{rec.case_id}` · NitroTrace v{res.engine_version}",
             "- Reviewer (role / affiliation): ______________________   Date: __________", "",
             "## A. Inputs", ""]
        items = []
        if rec.observed_product:
            items.append(("Observed ammonia", nt_value(rec.observed_product), rec.observed_product.evidence))
        for g in rec.gas_sources:
            items.append((f"{g.label} — impurity", nt_value(g.impurity) if g.impurity else "not measured",
                          g.impurity.evidence if g.impurity else g.evidence))
            if g.flow:
                items.append((f"{g.label} — flow", f"{g.flow.value:g} {g.flow.unit}", g.flow.evidence))
        for c in rec.controls:
            items.append((f"{c.label} (conditions: {c.conditions_match})", nt_value(c.measurement), c.measurement.evidence))
        for i, (name, val, ev) in enumerate(items, 1):
            L += [f"**A{i}. {name}:** {val}",
                  f"  Source: [{ev.source_id or '-'}] {ev.location or ''}" + (f" — “{ev.quote}”" if ev.quote else ""),
                  "  - [ ] correct   - [ ] incorrect   - [ ] cannot tell      Note: ______________________", ""]
        L += ["## B. Judgements NitroTrace made", ""]
        for i, c in enumerate(res.contributions, 1):
            L += [f"**B{i}. {c.label}** was treated as **{c.role.replace('_', ' ')}**"
                  + (f" because: {c.reason}" if c.reason else " (matched, valid bound)."),
                  "  - [ ] agree   - [ ] disagree      Note: ______________________", ""]
        for j, a in enumerate(res.assumptions, len(res.contributions) + 1):
            L += [f"**B{j}. Assumption:** {a}", "  - [ ] reasonable   - [ ] not reasonable      Note: ______________________", ""]
        L += ["## C. Outcome", "",
              f"NitroTrace's verdict: **{res.status_label}** — {res.headline}", "",
              "- [ ] I agree with this verdict given the evidence", "- [ ] I would reach a different verdict: ______________________", ""]
        if req and (req["unknown"] or req["shrink"]):
            L += ["NitroTrace's 'what would close the gap' thresholds:", ""]
            L += [f"- {s['text']}" for u in req["unknown"] for s in u["specs"]] + [f"- {s['text']}" for s in req["shrink"]]
            L += ["", "- [ ] these are sensible next measurements   - [ ] not sensible: ______________________", ""]
        L += ["## D. Anything NitroTrace missed?", "", "______________________________________________", ""]
        return "\n".join(L)

    def to_markdown(rec: ExperimentRecord, res: AuditResult) -> str:
        src = sources()
        L = [f"# NitroTrace audit — {rec.title}", "",
             f"- **Case:** `{rec.case_id}` ({rec.record_type})",
             f"- **Status:** {res.status_label}",
             f"- **Coverage:** " + ", ".join(f"{k}: {v}" for k, v in res.coverage.items()),
             f"- **Budget:** {res.budget_label}",
             f"- **Engine:** v{res.engine_version}", "",
             f"> {res.headline}", ""]
        if rec.description:
            L += [rec.description, ""]
        L += ["## Comparison (nmol N)", "", "| Quantity | Lower | Upper |", "|---|---:|---:|"]
        if res.observed_nmol:
            L.append(f"| Observed product | {nt_fmt(res.observed_nmol['lower'])} | {nt_fmt(res.observed_nmol['upper'])} |")
        L.append(f"| Valid contamination budget | {nt_fmt(res.budget_valid_nmol['lower'])} | {nt_fmt(res.budget_valid_nmol['upper'])} |")
        if res.budget_with_scenarios_nmol != res.budget_valid_nmol:
            L.append(f"| Budget incl. what-if measurements (not valid) | {nt_fmt(res.budget_with_scenarios_nmol['lower'])} | "
                     f"{nt_fmt(res.budget_with_scenarios_nmol['upper'])} |")
        L += ["", "## Explanation", ""] + [f"- {e}" for e in res.explanation] + [""]
        L += ["## Contributions", "", "| Source | Category | Role | Lower | Upper | Note |", "|---|---|---|---:|---:|---|"]
        for c in res.contributions:
            L.append(f"| {c.label} | {c.category} | {c.role} | {nt_fmt(c.lower_nmol)} | {nt_fmt(c.upper_nmol)} | {c.reason or ''} |")
        L += ["", "## Unquantified sources", ""]
        L += [f"- **{u.category}:** {u.reason}" for u in res.unquantified] or ["- None"]
        if res.feasible_allocation:
            fa = res.feasible_allocation
            L += ["", "## Illustrative feasible explanation", "", f"_{fa['label']}_", ""]
            L += [f"- {k}: {nt_fmt(v)} nmol N" for k, v in fa["allocation_nmol"].items()]
        L += ["", "## Calculation trail", ""]
        for step in res.observed_trail + [s for c in res.contributions for s in c.trail]:
            L.append(f"- **{step.label}**" + (f" — `{step.formula}`" if step.formula else ""))
            L += [f"  - {k}: {v}" for k, v in step.values.items()]
            if step.result:
                L.append(f"  - → {step.result}")
        L += ["", "## Assumptions", ""] + ([f"- {a}" for a in res.assumptions] or ["- None"])
        if res.issues:
            L += ["", "## Issues", ""] + [f"- [{i.severity}] `{i.code}` {i.message}" for i in res.issues]
        L += ["", "## Suggested next controls (protocol-based)", ""]
        for r in res.recommendations:
            L.append(f"### {r.title}")
            L += [f"_Trigger:_ {r.trigger}", "", r.text, ""]
            for ref in r.references:
                L.append(f"> “{ref['quote']}” — [{ref['source_id']}] {ref['location']}")
            L.append("")
        L += ["## Evidence", ""]
        ev_items = []
        if rec.observed_product:
            ev_items.append(("Observed product", rec.observed_product.evidence))
        for g in rec.gas_sources:
            ev_items.append((g.label, g.evidence))
            if g.impurity:
                ev_items.append((f"{g.label} — impurity", g.impurity.evidence))
            if g.flow:
                ev_items.append((f"{g.label} — flow", g.flow.evidence))
        for c in rec.controls:
            ev_items.append((c.label, c.measurement.evidence))
        for name, ev in ev_items:
            if not (ev.source_id or ev.quote or ev.location or ev.note):
                continue
            conf = "confirmed" if ev.confirmed else "unconfirmed"
            L.append(f"- **{name}** ({ev.provenance}, {conf}): [{ev.source_id or '-'}] {ev.location or ''} "
                     + (f"“{ev.quote}”" if ev.quote else "") + (f" — {ev.note}" if ev.note else ""))
        used = sorted({s for s in rec.sources} | {r['source_id'] for x in res.recommendations for r in x.references})
        L += ["", "## References", ""] + [f"- [{s}] {src[s]['citation']} {src[s].get('url','')}" for s in used if s in src]
        L += ["", "---", "", f"_{DISCLAIMER}_", ""]
        return "\n".join(L)

    # ======================= inversion =======================
    # Evidence-requirement inversion: what contamination remains unbounded, and how stringent would new
    # measurements have to be to close the gap?
    # 
    # The audit is solved backwards. Every threshold carries the assumptions it rests on; meeting a
    # threshold can move the verdict to "exceeds the quantified budget", which by itself still does not
    # demonstrate nitrogen reduction.

    N_OK = ("quantified", "not_applicable", "specified_synthetic")

    ALONE_SPLIT = ("'Alone' assumes every other unbounded category contributes nothing; 'equal split' shares the "
                   "headroom equally. Neither is a measurement.")
    NOT_PROOF = ("Meeting these thresholds can move the verdict to 'exceeds the quantified budget'. That alone does not "
                 "demonstrate nitrogen reduction: isotope-labelled and other controls are still required.")

    def obs_duration_s(rec: ExperimentRecord) -> float | None:
        op = rec.observed_product
        return op.interval.duration_min * 60 if op is not None and op.interval is not None else None

    def gas_amount_mol(src, rec: ExperimentRecord) -> float | None:
        """Moles of gas delivered through one source during the observation, or None if not determinable."""
        try:
            if src.total_volume_L is not None:
                return gas_moles_from_volume(src.total_volume_L, src.total_volume_temperature_K,
                                             src.total_volume_pressure_Pa)[0]
            if src.flow is None:
                return None
            dur = src.exposure.duration_min if src.exposure else observation_duration(rec)
            if dur is None:
                return None
            f = src.flow
            return gas_moles(f.value, f.unit, f.volume_basis, dur, f.temperature_K, f.pressure_Pa,
                             f.reference_temperature_K, f.reference_pressure_Pa)[0]
        except UnitError:
            return None

    def per_unit_nmol(m) -> float | None:
        """nmol N represented by 1 unit of a product measurement (the conversion is linear), or None."""
        try:
            return to_nmol_n(1.0, m.unit, m.species, m.basis, m.volume_mL, m.dilution_factor)[0]
        except UnitError:
            return None

    def nt_nh4_ug_per_L(nmol: float, volume_mL: float) -> float:
        return nmol * 1e-9 * MOLAR_MASS_G_PER_MOL["NH4+"] * 1e6 / (volume_mL / 1000)

    def evidence_requirements(rec: ExperimentRecord, res: AuditResult) -> dict | None:
        """Invert the audit: which measurements, at what level, would change the verdict?"""
        if res.status == "input_inconsistency" or res.observed_nmol is None:
            return None
        obs_lo = res.observed_nmol["lower"]
        U = res.budget_valid_nmol["upper"]
        headroom = obs_lo - U
        op = rec.observed_product
        unit_n = per_unit_nmol(op)
        t_s = obs_duration_s(rec)
        area = rec.context.electrode_area_cm2
        out = {
            "observed_lower_nmol": obs_lo, "valid_upper_nmol": U, "headroom_nmol": headroom,
            "floor_nmol": U, "floor_in_record_units": (U / unit_n) if unit_n else None,
            "record_unit": f"{op.unit} {op.species or ''}".strip(),
            "floor_rate_mol_s_cm2": (U * 1e-9 / (t_s * area)) if (t_s and area) else None,
            "unknown": [], "shrink": [], "notes": [], "caveats": [NOT_PROOF],
        }
        unknown_cats = [k for k, v in res.coverage.items() if v not in N_OK]

        if headroom > 0:
            share = headroom / max(len(unknown_cats), 1)
            if len(unknown_cats) > 1:
                out["caveats"].append(ALONE_SPLIT)
            for cat in unknown_cats:
                item = {"category": cat, "state": res.coverage[cat], "max_nmol_alone": headroom,
                        "max_nmol_equal_split": share, "specs": []}
                if cat == "gas":
                    for g in rec.gas_sources:
                        n = gas_amount_mol(g, rec)
                        partial = res.coverage[cat] == "partial_species" and g.impurity is not None
                        what = (f"Reactive-N species not yet bounded (beyond the measured {g.impurity.species}), as N,"
                                if partial else "Total reactive-N impurity (NOx + NH3 + N2O, as N)")
                        if n:
                            item["specs"].append({
                                "source": g.label, "source_id": g.id, "additional": partial, "kind": "gas",
                                "text": (f"{what} in the gas delivered to the cell ≤ {headroom / n:.3g} ppb (alone) or "
                                         f"≤ {share / n:.3g} ppb (equal split), for {n:.4g} mol of delivered gas."),
                                "ppb_alone": headroom / n, "ppb_split": share / n, "n_gas_mol": n,
                                "big": headroom / n, "big_unit": "ppb N in delivered gas (alone)",
                                "assumptions": [
                                    "η = 1: every delivered reactive-N molecule is assumed to end up as measured "
                                    "ammonia (conservative upper bound, not a measured efficiency).",
                                    "Delivered gas amount from the record's stated flow, reference conditions and "
                                    "exposure interval.",
                                    "Measured downstream of any purifier, under the experiment's own conditions."]})
                        else:
                            item["specs"].append({
                                "source": g.label, "source_id": g.id, "kind": "gas", "big": headroom,
                                "big_unit": "nmol N, x_N · n_gas (alone)",
                                "text": (f"Gas amount not determinable: state flow, reference conditions and exposure. "
                                         f"Requirement: x_N · n_gas ≤ {headroom:.3g} nmol."),
                                "assumptions": ["η = 1 (conservative)."]})
                elif cat == "electrolyte":
                    v = op.volume_mL
                    spec = {"source": "electrolyte", "kind": "electrolyte", "big": headroom,
                            "big_unit": "nmol N (alone)",
                            "text": f"Matched electrolyte / whole-system blank ≤ {headroom:.3g} nmol N (alone)",
                            "assumptions": ["Blank run with the same electrolyte batch, cell, volume and interval as "
                                            "the experiment (matched conditions)."]}
                    if v:
                        spec["big"], spec["big_unit"] = nt_nh4_ug_per_L(headroom, v), f"µg/L NH4+ in {v:g} mL (alone)"
                        spec["text"] += (f" = {nt_nh4_ug_per_L(headroom, v):.3g} µg/L NH4+ in {v:g} mL; "
                                         f"equal split: {nt_nh4_ug_per_L(share, v):.3g} µg/L.")
                    item["specs"].append(spec)
                else:
                    item["specs"].append({
                        "source": "components", "kind": "components", "big": headroom, "big_unit": "nmol N (alone)",
                        "text": (f"Matched release from cell components, electrode, membrane and ionomer ≤ "
                                 f"{headroom:.3g} nmol N (alone) / {share:.3g} nmol N (equal split)."),
                        "assumptions": ["Measured under the experiment's own conditions and interval (same electrode, "
                                        "potential history and electrolyte)."]})
                out["unknown"].append(item)
            if not unknown_cats:
                out["notes"].append(f"Every modelled category is bounded; the observation exceeds the valid budget by "
                                    f"{headroom:.3g} nmol N (×{obs_lo / U:.3g})." if U > 0 else
                                    "Every modelled category is bounded and the valid budget is zero.")
        else:
            deficit = -headroom
            out["notes"].append(f"The valid budget already reaches the observation (deficit {deficit:.3g} nmol N). "
                                "Unknown sources cannot change this; only tighter bounds on counted sources can.")
            for c in [c for c in res.contributions if c.included]:
                new_upper = c.upper_nmol - deficit
                row = {"source": c.label, "source_id": c.source_id, "category": c.category,
                       "current_upper_nmol": c.upper_nmol, "required_upper_nmol": new_upper, "feasible": new_upper > 0,
                       "assumptions": ["All other counted bounds stay as they are."]}
                if new_upper <= 0:
                    row["text"] = "Even a zero bound here is not enough: the other counted sources alone reach the observation."
                else:
                    g = next((g for g in rec.gas_sources if g.id == c.source_id), None)
                    ctl = next((k for k in rec.controls if k.id == c.source_id), None)
                    k = new_upper / c.upper_nmol
                    if g is not None and g.impurity is not None:
                        row["required_value"] = g.impurity.upper * k
                        row["big"], row["big_unit"] = row["required_value"], f"{g.impurity.unit} measured, delivered"
                        row["text"] = (f"Measured delivered-gas impurity ≤ {g.impurity.upper * k:.3g} {g.impurity.unit} "
                                       f"(now {g.impurity.upper:g} {g.impurity.unit}).")
                    elif ctl is not None and ctl.measurement.upper is not None:
                        row["required_value"] = ctl.measurement.upper * k
                        row["big"], row["big_unit"] = row["required_value"], ctl.measurement.unit
                        row["text"] = (f"Matched control ≤ {ctl.measurement.upper * k:.3g} {ctl.measurement.unit} "
                                       f"(now {ctl.measurement.upper:g} {ctl.measurement.unit}"
                                       f"{', i.e. a lower detection limit' if ctl.measurement.nondetect else ''}).")
                    else:
                        row["big"], row["big_unit"] = new_upper, "nmol N"
                        row["text"] = f"Upper bound ≤ {new_upper:.3g} nmol N."
                out["shrink"].append(row)
            if unknown_cats:
                out["notes"].append("Even after that, the verdict can only reach 'exceeds the quantified budget' once the "
                                    f"unbounded categories ({', '.join(unknown_cats)}) are also measured.")
            if unit_n:
                out["notes"].append(f"Alternatively the observation would need to be > {U / unit_n:.3g} {out['record_unit']} "
                                    f"({U:.3g} nmol N) with today's controls.")
        return out

    # ======================= planner =======================
    # Forward planning with the audit's conservative arithmetic: signal vs. contamination budget before a run,
    # and the feed-gas impurity level that alone would supply a reported rate.
    # 
    # Assumptions (stated wherever results are shown): constant true rate, constant flow, constant delivered
    # impurity level, capture fraction η (η = 1 is the conservative default), and a time-independent matched blank.

    REF_CONDITIONS = {"0 °C, 1 atm": (273.15, 101325.0), "20 °C, 1 atm": (293.15, 101325.0),
                      "25 °C, 1 atm": (298.15, 101325.0)}

    PLAN_ASSUMPTIONS = (
        "Constant true NRR rate, gas flow, delivered impurity level and capture fraction η over the run.",
        "η = 1 means every delivered reactive-N molecule is counted as product (conservative).",
        "The matched blank is a fixed amount, independent of run time.",
    )

    def planner(rate_mol_s_cm2: float, area_cm2: float, flow_sccm: float, ref_TP: tuple[float, float], ppb_N: float,
                eta: float, run_h: float, blank_ug_L: float, volume_mL: float, margin: float,
                fixed_volume_L: float | None = None, fixed_T_K: float = 298.15, fixed_P_Pa: float = 101325.0,
                n: int = 240) -> dict:
        """Signal and conservative contamination (nmol N) versus time, and the closed-form break-even time."""
        T_ref, P_ref = ref_TP
        s = rate_mol_s_cm2 * area_cm2 * 1e9                                   # signal, nmol N / s
        blank = blank_ug_L * (volume_mL / 1000) * 1e-6 / MOLAR_MASS_G_PER_MOL["NH4+"] * 1e9   # nmol N, fixed
        if fixed_volume_L:
            n_fixed = fixed_P_Pa * fixed_volume_L / 1000 / (R * fixed_T_K)
            g, gas_const = 0.0, n_fixed * ppb_N * eta                         # nmol N, capped
        else:
            ndot = P_ref * flow_sccm * 1e-6 / 60 / (R * T_ref)                # mol gas / s
            g, gas_const = ndot * ppb_N * eta, 0.0                            # nmol N / s
        b = blank + gas_const
        t = [run_h * 3600 * i / (n - 1) for i in range(n)]
        sig = [s * x for x in t]
        con = [g * x + b for x in t]
        # S(t) >= margin * C(t)  <=>  t (s - margin g) >= margin b
        if s <= margin * g:
            t_star, verdict = None, "never"
        else:
            t_star, verdict = margin * b / (s - margin * g), "after"
        return {"t_s": t, "signal": sig, "contamination": con, "s": s, "g": g, "b": b,
                "t_star_s": t_star, "verdict": verdict, "asymptotic_ratio": (s / g) if g > 0 else math.inf,
                "min_rate_mol_s_cm2": margin * (g + b / (run_h * 3600)) / (area_cm2 * 1e9),
                "end_ratio": (sig[-1] / con[-1]) if con[-1] > 0 else math.inf,
                "assumptions": list(PLAN_ASSUMPTIONS)}

    RATE_UNITS = {
        "mol s⁻¹ cm⁻²": lambda v, area, load: v * area,
        "nmol s⁻¹ cm⁻²": lambda v, area, load: v * 1e-9 * area,
        "µg(NH3) h⁻¹ cm⁻²": lambda v, area, load: v * 1e-6 / MOLAR_MASS_G_PER_MOL["NH3"] / 3600 * area,
        "µg(NH3) h⁻¹ mg_cat⁻¹": lambda v, area, load: v * 1e-6 / MOLAR_MASS_G_PER_MOL["NH3"] / 3600 * load * area,
    }
    RATE_UNIT_ALIASES = {"mol/s/cm2": "mol s⁻¹ cm⁻²", "nmol/s/cm2": "nmol s⁻¹ cm⁻²",
                         "ug/h/cm2": "µg(NH3) h⁻¹ cm⁻²", "ug/h/mg": "µg(NH3) h⁻¹ mg_cat⁻¹"}

    def stress_row(rate: float, unit: str, area_cm2: float, loading_mg_cm2: float, flow_sccm: float,
                   ref_TP: tuple[float, float]) -> dict:
        """Delivered-gas reactive-N mole fraction (ppb, η = 1) that alone supplies the reported rate."""
        T_ref, P_ref = ref_TP
        unit = RATE_UNIT_ALIASES.get(unit, unit)
        mol_s = RATE_UNITS[unit](rate, area_cm2, loading_mg_cm2 or 0.0)
        ndot = P_ref * flow_sccm * 1e-6 / 60 / (R * T_ref)
        return {"product_nmol_s": mol_s * 1e9, "gas_mol_s": ndot,
                "equivalent_ppb": mol_s / ndot * 1e9 if ndot else math.inf}

    # ======================= whatif =======================
    # Hypothetical measurements: show how one new measurement would change the budget and the verdict.
    # 
    # Every value added here carries provenance "hypothetical", so the audit marks its result as hypothetical
    # and reports never present it as evidence.

    KINDS = ("gas", "electrolyte", "electrolyte_nmol", "components")

    nt_NOTE = "HYPOTHETICAL measurement entered to explore its effect; not evidence."

    def add_hypothetical(d: dict, kind: str, value: float, source_id: str | None = None) -> dict:
        """Return a copy of record ``d`` with one hypothetical measurement added.

        gas:         total reactive N (as N) in the gas delivered to the cell, ``value`` in ppb, for ``source_id``
                     (default: the first gas source). Replaces that source's impurity bound.
        electrolyte: matched electrolyte blank, ``value`` in µg/L NH4+, same volume and interval as the sample.
        electrolyte_nmol: matched electrolyte blank as an amount, ``value`` in nmol N (no volume needed).
        components:  matched component release, ``value`` in nmol N, same interval as the sample.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if value < 0:
            raise ValueError("value must be non-negative")
        e = copy.deepcopy(d)
        ev = {"provenance": "hypothetical", "confirmed": False, "note": nt_NOTE}
        op = e.get("observed_product") or {}
        if kind == "gas":
            sources = e.get("gas_sources") or []
            if not sources:
                raise ValueError("the record has no gas source to bound")
            g = next((s for s in sources if s["id"] == source_id), None) if source_id else sources[0]
            if g is None:
                raise ValueError(f"no gas source with id '{source_id}'")
            g["impurity"] = {"species": "total reactive N (as N)", "nitrogen_atoms": 1, "unit": "ppb", "lower": 0,
                             "upper": value, "bound_type": "hard_limit", "measured_where": "delivered_to_cell",
                             "covers_all_reactive_nitrogen": True, "evidence": ev}
        elif kind == "electrolyte":
            if not op.get("volume_mL"):
                raise ValueError("the observed product has no volume; a blank concentration cannot become an amount")
            e.setdefault("controls", []).append({
                "id": "hypothetical_blank", "label": "Hypothetical matched electrolyte blank", "kind": "electrolyte_blank",
                "covers": ["electrolyte"], "conditions_match": "matched", "evidence": ev,
                "measurement": {"species": "NH4+", "basis": "concentration_mass", "unit": "µg/L", "lower": 0,
                                "upper": value, "bound_type": "hard_limit", "volume_mL": op["volume_mL"],
                                "interval": op.get("interval"), "evidence": ev}})
        elif kind == "electrolyte_nmol":
            e.setdefault("controls", []).append({
                "id": "hypothetical_blank", "label": "Hypothetical matched electrolyte blank", "kind": "electrolyte_blank",
                "covers": ["electrolyte"], "conditions_match": "matched", "evidence": ev,
                "measurement": {"species": "N", "basis": "amount", "unit": "nmol", "lower": 0, "upper": value,
                                "bound_type": "hard_limit", "interval": op.get("interval"), "evidence": ev}})
        else:
            e.setdefault("controls", []).append({
                "id": "hypothetical_components", "label": "Hypothetical matched component release",
                "kind": "component_release", "covers": ["components"], "conditions_match": "matched", "evidence": ev,
                "measurement": {"species": "N", "basis": "amount", "unit": "nmol", "lower": 0, "upper": value,
                                "bound_type": "hard_limit", "interval": op.get("interval"), "evidence": ev}})
        e["assumptions"] = list(e.get("assumptions", [])) + [f"{nt_NOTE} ({kind}: {value:g})"]
        return e

    def compare(d: dict, changes: list[tuple[str, float, str | None]]) -> dict:
        """Audit before and after one or more hypothetical measurements: [(kind, value, source_id), ...]."""
        before = run_audit(ExperimentRecord(**d))
        hyp = d
        for kind, value, source_id in changes:
            hyp = add_hypothetical(hyp, kind, value, source_id)
        after = run_audit(ExperimentRecord(**hyp))

        def brief(r):
            return {"status": r.status, "status_label": r.status_label, "coverage": r.coverage,
                    "valid_upper_nmol": r.budget_valid_nmol["upper"],
                    "observed_lower_nmol": r.observed_nmol["lower"] if r.observed_nmol else None,
                    "unquantified": [u.model_dump() for u in r.unquantified]}

        return {"changes": [list(c) for c in changes], "before": brief(before), "after": brief(after),
                "changed": before.status != after.status, "record": hyp, "after_result": after,
                "note": ("Hypothetical: shows what such measurements would imply. They are not evidence, and an "
                         "'exceeds' verdict would still not by itself demonstrate nitrogen reduction.")}

    def suggest(rec: ExperimentRecord, req: dict | None) -> list[tuple[str, float, str | None]]:
        """Hypothetical measurements at half of each 'alone' threshold: one per unbounded category."""
        if not req:
            return []
        r3 = lambda x: float(f"{x:.3g}")
        out = []
        for u in req["unknown"]:
            s = u["specs"][0]
            if u["category"] == "gas" and "ppb_split" in s:
                out.append(("gas", r3(s["ppb_split"] * 0.5), s.get("source_id")))
            elif u["category"] == "electrolyte" and "µg/L" in s.get("big_unit", ""):
                out.append(("electrolyte", r3(s["big"] / max(len(req["unknown"]), 1) * 0.5), None))
            elif u["category"] == "electrolyte":
                out.append(("electrolyte_nmol", r3(u["max_nmol_equal_split"] * 0.5), None))
            elif u["category"] == "components":
                out.append(("components", r3(u["max_nmol_equal_split"] * 0.5), None))
        return out

    # ======================= validation =======================
    # Benchmark against baselines, plus machine-checked properties run on every record.
    # 
    # Synthetic records test software and reasoning only; published records carry no ground-truth label
    # for whether N2 reduction occurred and are reported, not scored.

    def load_records(data: dict | None = None) -> dict[str, dict]:
        """{"cases/x.json" | "synthetic/y.json": record dict}. ``data`` may hold an embedded copy."""
        if data is not None:
            return {k: v for k, v in data.items() if k.startswith(("cases/", "synthetic/"))}
        out = {}
        for folder in ("cases", "synthetic"):
            for p in sorted((DATA_DIR / folder).glob("*.json")):
                out[f"{folder}/{p.name}"] = json.loads(p.read_text(encoding="utf-8"))
        return out

    def naive_status(d: dict) -> str:
        """Baseline: add every number entered, unknown = 0, ignore matching, coverage and unit checks."""
        try:
            op = d.get("observed_product")
            if not op:
                return "insufficient_information"
            lo = op.get("lower") if op.get("lower") is not None else 0.0
            obs_lo = to_nmol_n(lo, op["unit"], op.get("species") or "NH3", op.get("basis") or "amount",
                               op.get("volume_mL") or 1000, op.get("dilution_factor", 1))[0]
            total = 0.0
            for g in d.get("gas_sources", []):
                imp = g.get("impurity")
                if not imp or imp.get("upper") is None:
                    continue
                if g.get("total_volume_L"):
                    n = gas_moles_from_volume(g["total_volume_L"], g.get("total_volume_temperature_K") or 298.15,
                                              g.get("total_volume_pressure_Pa") or 101325)[0]
                elif g.get("flow"):
                    f = g["flow"]
                    dur = g.get("exposure") or op.get("interval") or {"end_min": 0}
                    dur = dur["end_min"] - dur.get("start_min", 0)
                    n = gas_moles(f["value"], f["unit"], "standard" if f["unit"] in ("sccm", "slpm") else "actual", dur,
                                  f.get("temperature_K") or 298.15, f.get("pressure_Pa") or 101325,
                                  f.get("reference_temperature_K") or f.get("temperature_K") or 298.15,
                                  f.get("reference_pressure_Pa") or 101325)[0]
                else:
                    continue
                total += n * gas_fraction(imp["upper"], imp["unit"]) * imp.get("nitrogen_atoms", 1) * 1e9
            for c in d.get("controls", []):
                m = c["measurement"]
                total += to_nmol_n(m["upper"], m["unit"], m.get("species") or "NH3", m.get("basis") or "amount",
                                   m.get("volume_mL") or 1000)[0]
            return "contamination_not_excluded" if total >= obs_lo else "exceeds_quantified_budget"
        except (UnitError, KeyError, TypeError):
            return "input_inconsistency"

    def checklist_status(d: dict) -> str:
        """Protocol-checklist baseline: 'gas analysis and a blank are present' -> trust the claim; no arithmetic."""
        if not d.get("observed_product"):
            return "insufficient_information"
        has_gas = any((g.get("impurity") or {}).get("upper") is not None for g in d.get("gas_sources", []))
        has_blank = any(c["kind"] in ("system_blank", "electrolyte_blank") for c in d.get("controls", []))
        return "exceeds_quantified_budget" if (has_gas and has_blank) else "insufficient_information"

    def prop_unit_invariance(d, res):
        if res.status == "input_inconsistency" or not res.observed_nmol:
            return None
        e = copy.deepcopy(d)
        e["observed_product"].update(species="N", basis="amount", unit="nmol", lower=res.observed_nmol["lower"],
                                     upper=res.observed_nmol["upper"], nondetect=False, volume_mL=None, dilution_factor=1.0)
        r2 = run_audit(ExperimentRecord(**e))
        return r2.status == res.status and math.isclose(r2.budget_valid_nmol["upper"], res.budget_valid_nmol["upper"],
                                                        rel_tol=1e-12, abs_tol=1e-12)

    def prop_unknown_never_zero(d, res):
        """Deleting any counted piece of evidence must never *create* an 'exceeds' verdict.

        Measurements set aside because they overlap are skipped: deleting one of two conflicting
        measurements removes the conflict itself, which legitimately changes the coverage."""
        set_aside = {c.source_id for c in res.contributions if c.role == "excluded_overlap"}
        checks = []
        for i, g in enumerate(d.get("gas_sources", [])):
            if g["id"] not in set_aside:
                e = copy.deepcopy(d); e["gas_sources"][i]["impurity"] = None; checks.append(e)
        for i, c in enumerate(d.get("controls", [])):
            if c["id"] not in set_aside and not set_aside:
                e = copy.deepcopy(d); e["controls"].pop(i); checks.append(e)
        if not checks:
            return None
        for e in checks:
            r2 = run_audit(ExperimentRecord(**e))
            if r2.status == "exceeds_quantified_budget" and res.status != "exceeds_quantified_budget":
                return False
        return True

    def prop_monotone(d, res):
        """Adding a matched component-release bound of k nmol raises the valid budget by exactly k (no system blank)."""
        if res.status == "input_inconsistency" or any(c["kind"] == "system_blank" for c in d.get("controls", [])):
            return None
        e = copy.deepcopy(d)
        iv = (d.get("observed_product") or {}).get("interval")
        e.setdefault("controls", []).append({"id": "probe", "label": "probe", "kind": "component_release",
                                             "covers": ["components"], "conditions_match": "matched",
                                             "measurement": {"species": "N", "basis": "amount", "unit": "nmol",
                                                             "lower": 0, "upper": 7.0, "interval": iv}})
        r2 = run_audit(ExperimentRecord(**e))
        return math.isclose(r2.budget_valid_nmol["upper"], res.budget_valid_nmol["upper"] + 7.0, rel_tol=1e-12)

    def prop_inversion_shrink(d, rec, res):
        """Apply each 'tighten' requirement at (1 ± 1e-6) and check the valid budget lands on the right side."""
        req = evidence_requirements(rec, res)
        if not req or not req["shrink"]:
            return None
        ok = True
        for row in req["shrink"]:
            if not row["feasible"] or "required_value" not in row:
                continue
            for f, want_below in ((1 - 1e-6, True), (1 + 1e-6, False)):
                e = copy.deepcopy(d)
                g = next((g for g in e.get("gas_sources", []) if g["id"] == row["source_id"]), None)
                if g is not None:
                    g["impurity"]["upper"] = row["required_value"] * f
                    g["impurity"]["lower"] = min(g["impurity"].get("lower") or 0, g["impurity"]["upper"])
                else:
                    c = next(c for c in e["controls"] if c["id"] == row["source_id"])
                    c["measurement"]["upper"] = row["required_value"] * f
                    if c["measurement"].get("lower") is not None:
                        c["measurement"]["lower"] = min(c["measurement"]["lower"], c["measurement"]["upper"])
                r2 = run_audit(ExperimentRecord(**e))
                ok &= (r2.budget_valid_nmol["upper"] < r2.observed_nmol["lower"]) == want_below
        return ok

    def prop_inversion_gas(d, rec, res):
        """Fill an unknown gas source at the 'alone' ppb requirement (1 ± 1e-6) and re-audit."""
        req = evidence_requirements(rec, res)
        if not req or len(req["unknown"]) != 1 or req["unknown"][0]["category"] != "gas":
            return None
        spec = next((s for s in req["unknown"][0]["specs"] if "ppb_alone" in s), None)
        if spec is None or spec["ppb_alone"] <= 0:
            return None
        if any(g.get("impurity") for g in d["gas_sources"] if g["id"] != spec["source_id"]):
            return None
        counted = next((c.upper_nmol for c in res.contributions if c.source_id == spec["source_id"] and c.included), 0.0)
        ok = True
        for f, want in ((1 - 1e-6, "exceeds_quantified_budget"), (1 + 1e-6, "contamination_not_excluded")):
            e = copy.deepcopy(d)
            g = next(g for g in e["gas_sources"] if g["id"] == spec["source_id"])
            g["impurity"] = {"species": "NOx+NH3 (as N)", "unit": "ppb", "lower": 0,
                             "upper": counted / spec["n_gas_mol"] + spec["ppb_alone"] * f,
                             "measured_where": "delivered_to_cell", "covers_all_reactive_nitrogen": True}
            g["eta_upper"], g["eta_basis"] = 1.0, "conservative_upper_bound_1"
            ok &= run_audit(ExperimentRecord(**e)).status == want
        return ok

    PROPERTIES = {
        "Unit invariance (any unit → same verdict & budget)": lambda d, rec, res: prop_unit_invariance(d, res),
        "Unknown is never zero (removing evidence never creates 'exceeds')": lambda d, rec, res: prop_unknown_never_zero(d, res),
        "Monotone budget (adding a bound adds exactly its amount)": lambda d, rec, res: prop_monotone(d, res),
        "Inversion round-trip: tighten requirements flip at the threshold": prop_inversion_shrink,
        "Inversion round-trip: gas ppb requirement flips the verdict": prop_inversion_gas,
    }

    def run_benchmark(records: dict[str, dict] | None = None) -> tuple[list[dict], dict]:
        records = records or load_records()
        rows, props = [], {k: [] for k in PROPERTIES}
        t0 = time.perf_counter()
        for k, d in records.items():
            rec = ExperimentRecord(**d)
            res = run_audit(rec)
            exp = d.get("expected") or {}
            ref = exp.get("budget_upper_nmol")
            rows.append({"case": d["case_id"], "type": d["record_type"], "expected": exp.get("status"),
                         "nitrotrace": res.status, "naive_sum": naive_status(d), "checklist": checklist_status(d),
                         "budget_upper_nmol": res.budget_valid_nmol["upper"],
                         "rel_error": abs(res.budget_valid_nmol["upper"] - ref) / ref if ref else None})
            for name, fn in PROPERTIES.items():
                v = fn(d, rec, res)
                if v is not None:
                    props[name].append((d["case_id"], v))
        syn = [r for r in rows if r["type"] == "synthetic"]
        false_exceeds = lambda col: sum(r[col] == "exceeds_quantified_budget" != r["expected"] for r in syn)
        errs = [r["rel_error"] for r in syn if r["rel_error"] is not None]
        summary = {
            "n_synthetic": len(syn),
            "nitrotrace_correct": sum(r["nitrotrace"] == r["expected"] for r in syn),
            "naive_correct": sum(r["naive_sum"] == r["expected"] for r in syn),
            "checklist_correct": sum(r["checklist"] == r["expected"] for r in syn),
            "false_exceeds_nitrotrace": false_exceeds("nitrotrace"),
            "false_exceeds_naive": false_exceeds("naive_sum"),
            "false_exceeds_checklist": false_exceeds("checklist"),
            "max_rel_error": max(errs) if errs else None, "n_numeric": len(errs),
            "elapsed_s": time.perf_counter() - t0,
            "properties": {k: (sum(v for _, v in vals), len(vals), [c for c, v in vals if not v]) for k, vals in props.items()},
        }
        return rows, summary


    return (
        ExperimentRecord,
        N_OK,
        STATUS_LABELS,
        compare,
        evidence_requirements,
        gas_amount_mol,
        gas_fraction,
        load_records,
        run_audit,
        run_benchmark,
        suggest,
        to_casefile,
        to_csv,
        to_json,
        to_markdown,
        to_review_pack,
    )


@app.cell(hide_code=True)
def evidence_agent():
    # Evidence agent: pasted methods/results text -> passages -> (optional) Claude candidates.
    # Candidates are never trusted: any quote that does not occur verbatim in the text is dropped,
    # and nothing enters an audit until a person types it into the record.

    CLAUDE_MODEL = "claude-opus-5-5"

    TOPICS = {
        "product": r"\b(NH3|NH4\+?|ammonia|ammonium)\b.{0,80}\b(yield|amount|concentration|produced|nmol|µg|ug|mg L|µM)\b",
        "gas_flow": r"\b(flow(?: rate)?|purg\w*|bubbl\w*|sccm|mL\s?min)\b",
        "duration": r"\b(\d+(?:\.\d+)?)\s?(h|hours?|min|minutes?)\b",
        "blank": r"\b(blank|background|control|open[- ]circuit|OCV|argon|Ar\b)",
        "impurity": r"\b(NOx|NO2?|N2O|nitrate|nitrite|ppm|ppb|impurit\w*|contaminat\w*)\b",
        "isotope": r"(15N|¹⁵N|isotop\w*|labell?ed)",
    }
    VALUE_PATTERNS = {
        "flow": r"(\d+(?:\.\d+)?)\s?(mL\s?min[−-]1|mL/min|sccm|L\s?min[−-]1)",
        "duration": r"(\d+(?:\.\d+)?)\s?(h|min)\b",
        "gas_fraction": r"(\d+(?:\.\d+)?)\s?(ppm|ppb|ppbmol|ppmv|mol\s?%)",
        "mass_conc": r"(\d+(?:\.\d+)?)\s?(µg|ug|ng|mg)\s?(L[−-]1|/L|mL[−-]1|/mL)",
        "amount": r"(\d+(?:\.\d+)?)\s?(nmol|µmol|umol|pmol)\b",
        "volume": r"(\d+(?:\.\d+)?)\s?mL\b(?!\s?min)",
    }


    def find_passages(text):
        out = []
        sentences = re.split(r"(?<=[.;])\s+", re.sub(r"\s+", " ", text))
        for i, sent in enumerate(sentences):
            topics = [t for t, rx in TOPICS.items() if re.search(rx, sent, re.I)]
            if not topics or (len(topics) < 2 and topics != ["isotope"]):
                continue
            vals = {k: [" ".join(m.groups()) for m in re.finditer(rx, sent)] for k, rx in VALUE_PATTERNS.items()}
            out.append({"sentence": i + 1, "topics": ", ".join(topics), "text": sent[:600],
                        "regex_values": "; ".join(f"{k}: {', '.join(v)}" for k, v in vals.items() if v)})
        return out


    CANDIDATE_SCHEMA = {
        "type": "object", "additionalProperties": False, "required": ["candidates"],
        "properties": {"candidates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["field", "value", "unit", "species", "quote"],
            "properties": {
                "field": {"type": "string", "enum": [
                    "observed_product", "electrolyte_volume", "collection_duration", "gas_flow", "gas_temperature",
                    "gas_pressure", "gas_impurity", "blank_result", "component_release", "detection_limit",
                    "isotope_evidence", "electrode_area"]},
                "value": {"type": ["number", "null"]}, "unit": {"type": ["string", "null"]},
                "species": {"type": ["string", "null"]}, "quote": {"type": "string"}}}}},
    }
    AGENT_SYSTEM = (
        "You extract candidate measurement values from an electrochemical ammonia-synthesis paper. Only report values "
        "explicitly written in the text, each with an exact verbatim supporting quote. If a value is not stated, use null "
        "- never supply a customary flow rate, electrode area, gas purity, temperature, pressure or control result. Keep "
        "the species exactly as written. A scientist reviews every candidate before use."
    )


    def claude_candidates(text, api_key):
        try:
            import anthropic
        except ImportError:
            return {"error": "The `anthropic` package is not installed in this kernel."}
        client = anthropic.Anthropic(api_key=api_key or None)
        msgs = [{"role": "user", "content": f"Paper text:\n\n{text[:150000]}"}]
        try:
            try:
                response = client.beta.messages.create(
                    model=CLAUDE_MODEL, max_tokens=16000, system=AGENT_SYSTEM, messages=msgs,
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                    output_config={"effort": "medium", "format": {"type": "json_schema", "schema": CANDIDATE_SCHEMA}})
            except (TypeError, anthropic.BadRequestError):
                response = client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=16000, system=AGENT_SYSTEM, messages=msgs,
                    output_config={"format": {"type": "json_schema", "schema": CANDIDATE_SCHEMA}})
        except anthropic.AuthenticationError:
            return {"error": "No valid Anthropic credentials: paste an API key above."}
        except anthropic.RateLimitError:
            return {"error": "Rate limited by the Anthropic API; try again shortly."}
        except anthropic.APIStatusError as e:
            return {"error": f"Anthropic API error {e.status_code}: {e.message}"}
        except anthropic.APIConnectionError:
            return {"error": "Could not reach the Anthropic API from this sandbox."}
        if response.stop_reason == "refusal":
            return {"error": "The model declined this request."}
        raw = "".join(b.text for b in response.content if b.type == "text")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "Model output was not valid JSON."}
        hay = re.sub(r"\s+", " ", text)
        kept, dropped = [], 0
        for c in parsed.get("candidates", []):
            if c.get("quote") and re.sub(r"\s+", " ", c["quote"]) in hay:
                kept.append(c | {"quote_verified": "✓ verbatim", "status": "unconfirmed"})
            else:
                dropped += 1
        return {"model": response.model, "candidates": kept, "dropped": dropped}


    return claude_candidates, find_passages


@app.cell(hide_code=True)
def bench_results(DATA, load_records, run_benchmark):
    # Benchmark + machine-checked properties, computed once at startup.
    bench_rows, bench_summary = run_benchmark(load_records(DATA))

    return (bench_summary,)


@app.cell(hide_code=True)
def ui_assets():
    # Front-end source for the NitroTrace interface (ES module + stylesheet), kept as data.
    NT_ESM = r'''// NitroTrace front end. Python (the marimo kernel) owns the engine; this file only draws and asks.
    const S = {
      contamination_not_excluded: { hex: "#c8401f", label: "Contamination not excluded",
        say: "The measured ammonia <em>fits inside</em> the contamination that was actually measured." },
      exceeds_quantified_budget: { hex: "#1d6b4f", label: "Exceeds the quantified budget",
        say: "The measured ammonia is <em>larger</em> than every measured contamination source combined." },
      insufficient_information: { hex: "#a86e12", label: "Insufficient information",
        say: "There is <em>not enough evidence</em> to say where this ammonia came from." },
      input_inconsistency: { hex: "#4f46a5", label: "Input inconsistency",
        say: "These inputs <em>cannot yet be put on one basis</em>." },
    };
    const COV = { quantified: ["Bounded", "#1d6b4f"], not_applicable: ["Not applicable", "#8a8474"],
      specified_synthetic: ["Specified (synthetic)", "#20507a"], partial_species: ["Partly bounded", "#a86e12"],
      scenario_only: ["What-if only", "#a86e12"], ambiguous_overlap: ["Ambiguous overlap", "#4f46a5"], unknown: ["Unknown", "#c8401f"] };
    const ROLE = { valid_bound: ["counted", "#17150f"], scenario: ["what-if", "#a86e12"], excluded_overlap: ["overlap", "#8a8474"],
      excluded_unmatched: ["not matched", "#8a8474"], excluded_invalid: ["not a measurement", "#8a8474"] };
    const TABS = [["audit", "Audit"], ["settle", "What would settle it"], ["design", "Design before you run"],
      ["stress", "Claim stress-test"], ["evidence", "Evidence agent"], ["report", "Report"], ["method", "Method & proof"]];
    let CW = 760;
    const R = 8.314462618, REF = { "0 °C": 273.15, "20 °C": 293.15, "25 °C": 298.15 };

    const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
    function num(x, p = 3) {
      if (x == null || Number.isNaN(x)) return "—";
      if (!Number.isFinite(x)) return "∞";
      if (x === 0) return "0";
      const a = Math.abs(x);
      if (a >= 1e5 || a < 1e-3) {
        const [m, e] = x.toExponential(p - 1).split("e");
        return `${m}×10${String(+e).split("").map((c) => SUP[c]).join("")}`;
      }
      return String(+x.toPrecision(p));
    }
    const hrs = (h) => (h < 1 ? `${num(h * 60, 2)} min` : `${num(h, 3)} h`);
    const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
    const pill = (st) => `<span class="nt-pill" style="color:${S[st].hex}"><i></i>${S[st].label}</span>`;
    function download(name, text, type) {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([text], { type }));
      a.download = name; document.body.appendChild(a); a.click(); a.remove();
    }
    function logTicks(lo, hi) {
      const out = [];
      for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
        for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo && v <= hi) out.push([v, m === 1]); }
      return out;
    }
    const tickLabel = (v) => {
      const e = Math.round(Math.log10(v));
      if (Math.abs(v / 10 ** e - 1) < 1e-9 && (e > 3 || e < -2)) return `10${String(e).split("").map((c) => SUP[c]).join("")}`;
      return num(v, 2);
    };

    function md(src) {
      const lines = src.split("\n"); let out = "", list = false, table = null;
      const inline = (s) => s.split(/(`[^`]+`)/).map((part) => part.startsWith("`") && part.endsWith("`") && part.length > 1
        ? `<code>${esc(part.slice(1, -1))}</code>`
        : esc(part).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|\s)_([^_]+?)_(?=$|[\s.,;:)])/g, "$1<i>$2</i>")).join("");
      const flush = () => {
        if (list) { out += "</ul>"; list = false; }
        if (table) {
          out += `<table class="nt-tbl"><tr>${table[0].map((c) => `<th>${inline(c)}</th>`).join("")}</tr>` +
            table.slice(2).map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("") + "</table>";
          table = null;
        }
      };
      for (const raw of lines) {
        const l = raw.trimEnd();
        if (l.startsWith("|")) { const cells = l.slice(1, -1).split("|").map((c) => c.trim()); (table ||= []).push(cells); continue; }
        if (/^\s*- /.test(l)) { if (table) flush(); if (!list) { out += "<ul>"; list = true; } out += `<li>${inline(l.replace(/^\s*- /, ""))}</li>`; continue; }
        flush();
        if (l.startsWith("### ")) out += `<h3>${inline(l.slice(4))}</h3>`;
        else if (l.startsWith("## ")) out += `<h2>${inline(l.slice(3))}</h2>`;
        else if (l.startsWith("# ")) out += `<h1>${inline(l.slice(2))}</h1>`;
        else if (l.startsWith("> ")) out += `<blockquote>${inline(l.slice(2))}</blockquote>`;
        else if (l === "---") out += "<hr>";
        else if (l) out += `<p>${inline(l)}</p>`;
      }
      flush(); return out;
    }

    /* ---------------------------------------------------------------- audit ---- */
    function ruler(a) {
      const rows = [];
      if (a.observed) rows.push({ k: "obs", label: "Measured ammonia", lo: a.observed.lower, hi: a.observed.upper, kind: "obs" });
      for (const c of a.contributions) rows.push({ label: c.label, lo: c.lower, hi: c.upper, kind: c.role, sub: ROLE[c.role][0] });
      rows.push({ label: "Valid contamination budget", lo: a.valid.lower, hi: a.valid.upper, kind: "budget", rule: true });
      if (a.with_scen.upper !== a.valid.upper) rows.push({ label: "…including what-ifs (not valid)", lo: a.with_scen.lower, hi: a.with_scen.upper, kind: "budget_scen" });
      const pos = rows.flatMap((r) => [r.lo, r.hi]).filter((v) => v > 0);
      let lo = pos.length ? Math.min(...pos) / 5 : 1e-2, hi = pos.length ? Math.max(...pos) * 5 : 10;
      if (hi / lo < 100) { const m = Math.sqrt(hi * lo); lo = m / 10; hi = m * 10; }
      const W = Math.max(520, CW), L = Math.min(300, W * 0.36), PR = W - 12, RH = 40, TOP = 34, H = TOP + rows.length * RH + 36;
      const x = (v) => L + (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (PR - L);
      const color = { obs: "#1d6b4f", valid_bound: "#17150f", scenario: "#a86e12", excluded_overlap: "#8a8474",
        excluded_unmatched: "#8a8474", excluded_invalid: "#8a8474", budget: "#20507a", budget_scen: "#a86e12" };
      let g = `<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="#a86e12" stroke-width="2.2"/></pattern>`;
      for (const [k, c] of Object.entries(color))
        g += `<linearGradient id="z-${k}" x1="0" x2="1"><stop offset="0" stop-color="${c}" stop-opacity="0"/><stop offset=".12" stop-color="${c}"/></linearGradient>`;
      g += "</defs>";
      for (const [v, major] of logTicks(lo, hi)) {
        g += `<line x1="${x(v)}" x2="${x(v)}" y1="${TOP - 6}" y2="${H - 30}" stroke="${major ? "#d8d1c1" : "#e6e0d2"}"/>`;
        if (major || hi / lo < 3e3) g += `<text class="tick" x="${x(v)}" y="${H - 14}" text-anchor="middle">${tickLabel(v)}</text>`;
      }
      rows.forEach((r, i) => {
        const y = TOP + i * RH, cy = y + RH / 2, c = color[r.kind];
        if (r.rule) g += `<line x1="0" x2="${PR}" y1="${y + 1}" y2="${y + 1}" stroke="#17150f"/>`;
        g += `<text class="lab" x="0" y="${cy - 2}" ${r.kind === "obs" || r.kind === "budget" ? 'font-weight="600"' : ""}>${esc(r.label.length > L / 6.6 ? r.label.slice(0, Math.floor(L / 6.6) - 1) + "…" : r.label)}</text>`;
        g += `<text class="sub" x="0" y="${cy + 12}">[${num(r.lo)}, ${num(r.hi)}] nmol N${r.sub ? " · " + r.sub : ""}</text>`;
        const x0 = r.lo > 0 ? x(r.lo) : L, x1 = Math.max(x(r.hi), x0 + 3), h = r.kind === "obs" || r.kind === "budget" ? 14 : 10;
        const delay = `style="animation-delay:${i * 45}ms"`;
        if (r.hi <= 0) { g += `<text class="sub" x="${L + 4}" y="${cy + 4}">0 — nothing counted</text>`; return; }
        if (r.kind.startsWith("excluded")) g += `<rect class="nt-grow" ${delay} x="${x0}" y="${cy - h / 2}" width="${x1 - x0}" height="${h}" fill="none" stroke="${c}" stroke-dasharray="3 3"/>`;
        else if (r.kind === "scenario" || r.kind === "budget_scen") g += `<rect class="nt-grow" ${delay} x="${x0}" y="${cy - h / 2}" width="${x1 - x0}" height="${h}" fill="url(#hatch)" stroke="${c}" stroke-width=".8"/>`;
        else g += `<rect class="nt-grow" ${delay} x="${x0}" y="${cy - h / 2}" width="${x1 - x0}" height="${h}" fill="${r.lo > 0 ? c : `url(#z-${r.kind})`}"/>`;
        if (r.lo === r.hi) g += `<path d="M${x1} ${cy - 8} l6 8 l-6 8 l-6 -8z" fill="${r.kind.startsWith("excluded") ? "#f2efe7" : c}" stroke="${c}"/>`;
      });
      if (a.observed) {
        const ox = x(a.observed.lower);
        g += `<line x1="${ox}" x2="${ox}" y1="${TOP - 10}" y2="${H - 30}" stroke="#1d6b4f" stroke-dasharray="4 3"/>
              <text x="${ox}" y="${TOP - 16}" text-anchor="middle" class="sub" style="fill:#1d6b4f">measured lower bound</text>`;
      }
      return `<div class="nt-ruler"><svg viewBox="0 0 ${W} ${H}">${g}</svg></div>`;
    }

    function track(model, a) {
      const wrap = el(`<div class="nt-track" tabindex="0"></div>`);
      const vm = a.vmap || [];
      const pos = (k) => ((Math.log10(k) + 3) / 6) * 100;
      vm.forEach(([k, st], i) => {
        const l = i ? (pos(vm[i - 1][0]) + pos(k)) / 2 : 0, r = i < vm.length - 1 ? (pos(k) + pos(vm[i + 1][0])) / 2 : 100;
        wrap.appendChild(el(`<div class="seg" style="left:${l}%;width:${r - l + 0.05}%;background:${S[st].hex}"></div>`));
      });
      for (let e = -3; e <= 3; e++) wrap.appendChild(el(`<div class="ax" style="left:${pos(10 ** e)}%">×${num(10 ** e)}</div>`));
      const thumb = el(`<div class="thumb"></div>`); wrap.appendChild(thumb);
      const statusAt = (k) => vm.reduce((b, v) => (Math.abs(Math.log10(v[0] / k)) < Math.abs(Math.log10(b[0] / k)) ? v : b), vm[0] || [1, a.status]);
      const place = (k) => { thumb.style.left = pos(k) + "%"; thumb.dataset.v = `×${num(k)} · ${S[statusAt(k)[1]].label.toLowerCase()}`; };
      place(a.scale || 1);
      const kFromX = (cx) => { const r = wrap.getBoundingClientRect(); const f = Math.min(1, Math.max(0, (cx - r.left) / r.width)); return 10 ** (f * 6 - 3); };
      let drag = false, last = a.scale || 1;
      const commit = (k) => { const snap = statusAt(k)[0]; place(snap); model.set("scale", snap); model.save_changes(); };
      wrap.addEventListener("pointerdown", (e) => { drag = true; wrap.setPointerCapture(e.pointerId); last = kFromX(e.clientX); place(last); });
      wrap.addEventListener("pointermove", (e) => { if (drag) { last = kFromX(e.clientX); place(last); } });
      wrap.addEventListener("pointerup", () => { if (drag) { drag = false; commit(last); } });
      wrap.addEventListener("keydown", (e) => {
        const i = vm.findIndex((v) => v[0] === statusAt(last)[0]);
        if (e.key === "ArrowRight" && i < vm.length - 1) { last = vm[i + 1][0]; commit(last); }
        if (e.key === "ArrowLeft" && i > 0) { last = vm[i - 1][0]; commit(last); }
      });
      return wrap;
    }

    function auditPage(model, a) {
      if (!a || !a.status) return el(`<div class="nt-main"><p class="nt-empty">Loading…</p></div>`);
      const m = el(`<div class="nt-main nt-fade"></div>`);
      const st = S[a.status];
      m.innerHTML = `
        <div class="nt-kicker">${pill(a.status)}<span class="caps">${esc(a.record_type)}</span><span class="mono cite">${esc(a.case_id)}</span>
          ${a.scale !== 1 ? `<span class="caps" style="color:#a86e12">what-if: signal ×${num(a.scale)}</span>` : ""}</div>
        <h1 class="nt-verdict">${st.say}</h1>
        <p class="nt-head">${esc(a.headline)}</p>
        <p class="nt-title">${esc(a.title)}${a.sources.length ? ` — <span class="cite">${a.sources.map(esc).join(" · ")}</span>` : ""}</p>
        ${a.hypothetical ? `<div class="nt-hyp">This record contains <b>hypothetical</b> inputs. The result shows what they would imply; it is not evidence.</div>` : ""}
        ${a.error ? `<div class="nt-err">${esc(a.error)}</div>` : ""}`;
      const sec = (title, note, body) => { const s = el(`<section class="nt-sec"><header><h3>${title}</h3><span class="caps">${note}</span></header></section>`); if (typeof body === "string") s.insertAdjacentHTML("beforeend", body); else s.appendChild(body); return s; };
      m.appendChild(sec("The comparison", "nmol of ammonia-equivalent N · log scale", ruler(a)));
      const t = el(`<div></div>`);
      t.appendChild(el(`<p style="margin:0;color:var(--ink-2);font-size:13px">Drag along the band to rescale the measured ammonia. The colour under every point is the verdict NitroTrace would reach there, with all evidence held fixed.</p>`));
      t.appendChild(track(model, a));
      m.appendChild(sec("If the signal were different", "verdict map · ×10⁻³ to ×10³", t));
      m.appendChild(sec("Coverage", a.budget_label, `<div class="nt-cols3">${Object.entries(a.coverage).map(([k, v]) => {
        const [lab, c] = COV[v] || [v, "#8a8474"]; const u = a.unquantified.find((q) => q.category === k);
        return `<div class="nt-cov"><span class="caps">${k}</span><div class="v">${lab}</div><div class="bar" style="background:${c}"></div><p>${esc(u ? u.reason : "Bounded by matched, valid measurements.")}</p></div>`; }).join("")}</div>`));
      m.appendChild(sec("Reasoning", "", `<div class="nt-cols"><div><span class="caps">Why this verdict</span><ol class="nt-list">${a.explanation.map((e) => `<li>${esc(e)}</li>`).join("")}</ol></div>
        <div><span class="caps">Never treated as zero</span><ol class="nt-list">${a.unquantified.map((u) => `<li><b>${esc(u.category)}</b> — ${esc(u.reason)}</li>`).join("") || '<li class="nt-empty">Every category is bounded.</li>'}</ol></div></div>`));
      m.appendChild(sec("Contributions", `${a.contributions.length} sources`, a.contributions.length ? `<table class="nt-tbl"><tr><th>Source</th><th>Category</th><th>Role</th><th style="text-align:right">Lower</th><th style="text-align:right">Upper</th><th>Why</th></tr>
        ${a.contributions.map((c) => `<tr class="${c.included ? "" : "off"}"><td>${esc(c.label)}</td><td>${esc(c.category)}</td><td><span class="nt-role" style="color:${ROLE[c.role][1]}">${ROLE[c.role][0]}</span></td><td class="n">${num(c.lower)}</td><td class="n">${num(c.upper)}</td><td style="font-size:12px;color:var(--ink-2)">${esc(c.reason || "")}</td></tr>`).join("")}</table>` : `<p class="nt-empty">No source could be bounded.</p>`));
      m.appendChild(sec("Next controls", "each quotes the protocol it comes from", a.recs.map((r, i) => `<div class="nt-proto"><div class="k">${String(i + 1).padStart(2, "0")}</div><div>
        <h4>${esc(r.title)}</h4><p><span class="caps">Trigger</span> ${esc(r.trigger)}</p><p>${esc(r.text)}</p>
        ${r.refs.map((q) => `<div class="quote">“${esc(q.quote)}”</div><div class="cite">[${esc(q.source_id)}] ${esc(q.location)}</div>`).join("")}</div></div>`).join("") || `<p class="nt-empty">None triggered.</p>`));
      const det = el(`<section class="nt-sec"></section>`);
      det.innerHTML = `
        <details><summary>Calculation trail</summary><div class="trail">${a.trail.map((s) => `<b>${esc(s.label)}${s.formula ? ` — <span class="mono">${esc(s.formula)}</span>` : ""}</b>${Object.entries(s.values).map(([k, v]) => `<div>${esc(k)}: ${esc(v)}</div>`).join("")}${s.result ? `<div>→ ${esc(s.result)}</div>` : ""}`).join("")}</div></details>
        <details><summary>Assumptions (${a.assumptions.length})</summary><ol class="nt-list">${a.assumptions.map((x) => `<li>${esc(x)}</li>`).join("") || "<li>None</li>"}</ol></details>
        <details><summary>Issues (${a.issues.length})</summary><ol class="nt-list">${a.issues.map((i) => `<li><span class="mono">${esc(i.severity)} · ${esc(i.code)}</span> — ${esc(i.message)}</li>`).join("") || "<li>None</li>"}</ol></details>
        <details class="edit"><summary>Edit the record</summary><p style="font-size:12.5px;color:var(--ink-2);margin:0 0 8px">Every value carries species, basis, unit, bound type and evidence. Missing means unknown.</p>
          <textarea class="nt-field" rows="18"></textarea><div style="margin:10px 0 16px"><button class="nt-btn">Re-run audit</button><button class="nt-btn ghost">Reset</button></div></details>`;
      const ta = det.querySelector("textarea"); ta.value = model.get("record_json");
      const [run, reset] = det.querySelectorAll(".edit button");
      run.onclick = () => { model.set("edit_json", { text: ta.value, n: Date.now() }); model.save_changes(); };
      reset.onclick = () => { model.set("selected", ""); model.save_changes(); model.set("selected", a.key); model.save_changes(); };
      m.appendChild(det);
      return m;
    }

    /* --------------------------------------------------------------- settle ---- */
    function settlePage(model, a) {
      const m = el(`<div class="nt-main nt-fade"></div>`);
      if (!a || !a.status) return m;
      const r = a.req;
      if (!r) { m.innerHTML = `<div class="nt-kicker">${pill(a.status)}</div><h1 class="nt-verdict">Nothing to invert <em>yet</em>.</h1><p class="nt-head">The audit needs an observed product and inputs without errors before it can be solved backwards.</p>${a.error ? `<div class="nt-err">${esc(a.error)}</div>` : ""}`; return m; }
      const h = r.headroom_nmol, unknown = r.unknown.length;
      const say = h > 0 ? (unknown ? `For the signal to exceed the contamination budget, everything still unbounded must stay below <em>${num(h)} nmol N</em> in total.`
        : `Every modelled source is bounded, and the signal clears them by <em>×${num(r.observed_lower_nmol / Math.max(r.valid_upper_nmol, 1e-300))}</em>.`)
        : `The counted contamination already reaches the signal. <em>Something measured has to come down.</em>`;
      m.innerHTML = `<div class="nt-kicker">${pill(a.status)}<span class="caps">evidence-requirement inversion</span></div>
        <h1 class="nt-verdict">${say}</h1>
        <p class="nt-head">The audit solved backwards: which measurement, at what level, would change the verdict, and what each threshold assumes. Each threshold is checked automatically: applied at ±10⁻⁶ it flips the verdict on the right side. Meeting them would not by itself demonstrate nitrogen reduction.</p>
        <section class="nt-sec"><header><h3>The gap</h3><span class="caps">nmol N</span></header>
          <div class="nt-gap"><div><span class="caps">Measured, lower bound</span><b>${num(r.observed_lower_nmol)}</b><span>what the experiment shows at least</span></div>
          <div><span class="caps">Valid budget, upper bound</span><b>${num(r.valid_upper_nmol)}</b><span>counted, matched contamination at most</span></div>
          <div><span class="caps">Headroom</span><b style="color:${h > 0 ? "#1d6b4f" : "#c8401f"}">${h > 0 ? "+" : ""}${num(h)}</b><span>room left for every source not yet measured</span></div></div>
          <div class="nt-note"><b>Contamination floor: ${num(r.floor_nmol)} nmol N${r.floor_in_record_units != null ? ` (${num(r.floor_in_record_units)} ${esc(r.record_unit)})` : ""}.</b>
            The smallest observation today's controls could ever call "above budget".${r.floor_nmol === 0 ? " It is zero because nothing matched and valid is counted yet: any signal beats an empty budget, which is exactly why that comparison proves nothing." : ""}
            ${r.floor_rate_mol_s_cm2 ? ` As a rate: ${num(r.floor_rate_mol_s_cm2)} mol s⁻¹ cm⁻².` : ""}</div></section>`;
      const specs = [];
      for (const u of r.unknown) for (const s of u.specs)
        specs.push({ cat: u.category, state: u.state, title: { gas: "Measure reactive nitrogen in the gas that reaches the cell", electrolyte: "Run a matched electrolyte / whole-system blank", components: "Measure release from cell components under the experiment's conditions" }[u.category], text: s.text, big: s.big, unit: s.big_unit, as: s.assumptions || [] });
      for (const s of r.shrink) specs.push({ cat: "tighten", title: `Tighten: ${s.source}`, text: s.text, big: s.feasible ? s.big : null, unit: s.feasible ? s.big_unit : "not possible alone", as: s.assumptions || [] });
      const sec = el(`<section class="nt-sec"><header><h3>${specs.length ? "Specifications" : "Nothing left to specify"}</h3><span class="caps">${unknown ? `${unknown} unbounded categor${unknown > 1 ? "ies" : "y"} · alone / equal split` : ""}</span></header></section>`);
      specs.forEach((s, i) => sec.appendChild(el(`<div class="nt-spec"><div class="k">${String(i + 1).padStart(2, "0")}</div><div><span class="caps">${esc(s.cat)}${s.state ? " · " + esc(s.state.replace(/_/g, " ")) : ""}</span><h4>${esc(s.title)}</h4><p>${esc(s.text)}</p>${s.as.length ? `<ul class="nt-as">${s.as.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}</div>
        <div class="big">${s.big != null ? `<b>≤ ${num(s.big)}</b>` : `<b style="color:#c8401f">—</b>`}<span>${esc(s.unit || "")}</span></div></div>`)));
      m.appendChild(sec);
      if (r.notes.length || r.caveats.length) m.appendChild(el(`<section class="nt-sec">${r.notes.map((n) => `<div class="nt-note">${esc(n)}</div>`).join("")}${r.caveats.map((n) => `<div class="nt-note caveat">${esc(n)}</div>`).join("")}</section>`));
      m.appendChild(whatifPanel(model, a));
      return m;
    }

    function whatifPanel(model, a) {
      const fields = a.whatif_fields || [];
      const sec = el(`<section class="nt-sec"><header><h3>Try hypothetical measurements</h3><span class="caps">not evidence · shows what they would imply</span></header>
        <p style="font-size:13.5px;color:var(--ink-2);margin:0 0 12px;max-width:860px">Enter measurements you could make. NitroTrace adds them as labelled <i>hypothetical</i> inputs and re-runs the audit. Values are prefilled at half of each equal-split threshold.</p>
        <div class="nt-wi"></div><div style="margin-top:12px"><button class="nt-btn run">Re-run with these</button></div><div class="wo"></div></section>`);
      const box = sec.querySelector(".nt-wi");
      fields.forEach((f, i) => {
        const row = el(`<label class="nt-wirow ${f.disabled ? "off" : ""}"><input type="checkbox" ${f.on && !f.disabled ? "checked" : ""} ${f.disabled ? "disabled" : ""}>
          <span>${esc(f.label)}${f.disabled ? `<small>${esc(f.disabled)}</small>` : ""}</span>
          <span class="u">≤</span><input class="nt-field v" type="number" min="0" step="any" value="${f.value ?? ""}" ${f.disabled ? "disabled" : ""}><span class="u">${esc(f.unit)}</span></label>`);
        row.dataset.i = i; box.appendChild(row);
      });
      sec.querySelector(".run").onclick = () => {
        const changes = [];
        box.querySelectorAll(".nt-wirow").forEach((row) => {
          const f = fields[+row.dataset.i], on = row.querySelector("input[type=checkbox]").checked, v = row.querySelector(".v").value;
          if (on && v !== "") changes.push({ kind: f.kind, value: +v, source_id: f.source_id || null });
        });
        model.set("whatif_req", { changes, n: Date.now() }); model.save_changes();
        sec.querySelector(".wo").innerHTML = `<p class="nt-empty">Re-running…</p>`;
      };
      const show = () => {
        const w = model.get("whatif_res") || {}, out = sec.querySelector(".wo");
        if (w.error) { out.innerHTML = `<div class="nt-err" style="margin-top:14px">${esc(w.error)}</div>`; return; }
        if (!w.before) { out.innerHTML = ""; return; }
        const cov = Object.keys(w.before.coverage).filter((k) => w.before.coverage[k] !== w.after.coverage[k])
          .map((k) => `<li><b>${esc(k)}</b>: ${esc(w.before.coverage[k].replace(/_/g, " "))} → ${esc(w.after.coverage[k].replace(/_/g, " "))}</li>`).join("");
        out.innerHTML = `<div class="nt-hyp" style="margin-top:16px"><span class="caps" style="color:inherit">hypothetical result</span>
          <div class="nt-ba">${pill(w.before.status)}<span class="arr">→</span>${pill(w.after.status)}</div>
          <div class="mono" style="font-size:13px">valid budget ${num(w.before.valid_upper_nmol)} → <b>${num(w.after.valid_upper_nmol)}</b> nmol N · measured lower bound ${num(w.after.observed_lower_nmol)} nmol N</div>
          ${cov ? `<ul class="nt-as" style="margin-top:8px">${cov}</ul>` : ""}
          ${w.after.unquantified.map((u) => `<div class="nt-note">Still unbounded — <b>${esc(u.category)}</b>: ${esc(u.reason)}</div>`).join("")}
          <p style="font-size:12.5px;margin:8px 0 0">${esc(w.note)}</p></div>`;
      };
      model.on("change:whatif_res", show); show();
      return sec;
    }

    /* --------------------------------------------------------------- design ---- */
    const D = { mode: "flow", rate: -11, area: 1, flow: Math.log10(20), ppb: 2, eta: 1, hours: 4, blank: 2, vol: 30, margin: 3, fixedL: 0.5, ref: "0 °C" };
    function plan(p, override = {}) {
      const q = { ...p, ...override };
      const T = REF[q.ref], s = 10 ** q.rate * q.area * 1e9, ppb = 10 ** q.ppb, flow = 10 ** q.flow;
      const blank = q.blank * (q.vol / 1000) * 1e-6 / 18.038 * 1e9;
      let g = 0, b = blank;
      if (q.mode === "flow") g = (101325 * flow * 1e-6 / 60 / (R * T)) * ppb * q.eta;
      else b += (101325 * q.fixedL / 1000 / (R * 298.15)) * ppb * q.eta;
      const Ts = q.hours * 3600, tstar = s <= q.margin * g ? null : (q.margin * b) / (s - q.margin * g);
      return { s, g, b, Ts, tstar, end: (s * Ts) / (g * Ts + b), rmin: (q.margin * (g + b / Ts)) / (q.area * 1e9) };
    }
    function designPage(model) {
      const m = el(`<div class="nt-main nt-fade"><div class="nt-kicker"><span class="caps">forward planning · same arithmetic as the audit</span></div>
        <h1 class="nt-answer"></h1><p class="nt-head sub2"></p>
        <section class="nt-sec"><header><h3>Plan the experiment</h3><span class="caps">everything updates as you move</span></header>
        <div class="nt-design"><div class="ctl"></div><div><div class="nt-chart c1"></div><div style="height:18px"></div>
          <span class="caps maplab"></span><div class="nt-map"><canvas width="260" height="150"></canvas><svg></svg></div>
          <p style="font-size:12.5px;color:var(--ink-2)">Click anywhere on the map to move the design there. Green: the signal beats the margin by the end of the run; red: it doesn't. Ink line: exactly the margin.</p></div></div></section>
        <section class="nt-sec"><header><h3>Why run time can't rescue a purged cell</h3><span class="caps">closed form</span></header>
          <div class="nt-cols"><p style="font-size:14px;color:var(--ink-2);margin:0">With gas flowing continuously, both the catalyst's ammonia and the feed gas's reactive nitrogen accumulate linearly in time, so the ratio settles at <span class="mono">S/C → r·A·R·T / (P·Q·x·η)</span>. It doesn't depend on how long you run. Only lower flow, cleaner gas or a closed gas volume moves it. With a fixed gas volume, gas contamination is capped and the ratio grows with time.</p>
          <p style="font-size:14px;color:var(--ink-2);margin:0"><b>Assumes</b> a constant true rate, flow, delivered impurity level and capture fraction η, and a fixed matched blank. Break-even time for a required margin <span class="mono">m</span>: <span class="mono">t* = m·b / (s − m·g)</span>, where <span class="mono">s</span> is the signal rate, <span class="mono">g</span> the gas contamination rate and <span class="mono">b</span> the fixed blank. If <span class="mono">s ≤ m·g</span> there is no t*. η = 1 is a labelled conservative scenario.</p></div></section></div>`);
      const ctl = m.querySelector(".ctl");
      const seg = el(`<div><div class="nt-seg mode"><button data-v="flow">Continuous purge</button><button data-v="fixed">Fixed gas volume</button></div><br><div class="nt-seg ref">${Object.keys(REF).map((k) => `<button data-v="${k}">${k} ref.</button>`).join("")}</div></div>`);
      ctl.appendChild(seg);
      const sliders = [
        ["rate", "Expected true NRR rate", -14, -7, 0.05, (v) => `${num(10 ** v)} mol s⁻¹ cm⁻²`],
        ["area", "Electrode area", 0.1, 25, 0.1, (v) => `${num(v)} cm²`],
        ["flow", "Gas flow", 0, 2.7, 0.01, (v) => `${num(10 ** v)} sccm`, "flow"],
        ["fixedL", "Fixed gas volume", 0.05, 5, 0.05, (v) => `${num(v)} L`, "fixed"],
        ["ppb", "Reactive N in delivered gas", -1, 4, 0.05, (v) => `${num(10 ** v)} ppb`],
        ["eta", "η, fraction captured", 0.01, 1, 0.01, (v) => num(v)],
        ["hours", "Run time", 0.25, 48, 0.25, (v) => hrs(v)],
        ["blank", "Matched blank, upper bound", 0, 50, 0.1, (v) => `${num(v)} µg/L NH₄⁺`],
        ["vol", "Electrolyte volume", 5, 200, 1, (v) => `${num(v)} mL`],
        ["margin", "Required margin", 1, 10, 0.5, (v) => `×${num(v)}`],
      ];
      const rows = {};
      for (const [k, lab, a, b, st, f, only] of sliders) {
        const r = el(`<div class="nt-ctl"><div class="row"><label>${lab}</label><span class="val"></span></div><input type="range" min="${a}" max="${b}" step="${st}" value="${D[k]}"></div>`);
        r.dataset.only = only || ""; rows[k] = r; ctl.appendChild(r);
        r.querySelector("input").addEventListener("input", (e) => { D[k] = +e.target.value; update(); });
      }
      seg.querySelectorAll(".mode button").forEach((b) => b.onclick = () => { D.mode = b.dataset.v; update(); });
      seg.querySelectorAll(".ref button").forEach((b) => b.onclick = () => { D.ref = b.dataset.v; update(); });
      const cv = m.querySelector("canvas"), ov = m.querySelector(".nt-map svg");
      cv.parentElement.addEventListener("click", (e) => {
        const r = cv.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width, fy = 1 - (e.clientY - r.top) / r.height;
        if (D.mode === "flow") D.flow = +(fx * 2.7).toFixed(2); else D.fixedL = +(0.05 + fx * 4.95).toFixed(2);
        D.ppb = +(-1 + fy * 5).toFixed(2);
        rows[D.mode === "flow" ? "flow" : "fixedL"].querySelector("input").value = D.mode === "flow" ? D.flow : D.fixedL;
        rows.ppb.querySelector("input").value = D.ppb; update();
      });
      function update() {
        seg.querySelectorAll(".mode button").forEach((b) => b.classList.toggle("on", b.dataset.v === D.mode));
        seg.querySelectorAll(".ref button").forEach((b) => b.classList.toggle("on", b.dataset.v === D.ref));
        for (const [k, , , , , f] of sliders) { rows[k].querySelector(".val").textContent = f(D[k]); rows[k].style.display = rows[k].dataset.only && rows[k].dataset.only !== D.mode ? "none" : ""; }
        const p = plan(D), H = D.hours;
        const ans = m.querySelector(".nt-answer"), sub = m.querySelector(".sub2");
        if (p.tstar === null) { ans.innerHTML = `<em>Not under these assumptions.</em> The feed gas supplies nitrogen ${num(p.g / p.s)}× as fast as the catalyst would fix it; with constant flow, purity and η, running longer doesn't change that ratio.`; }
        else if (p.tstar / 3600 > H) { ans.innerHTML = `Not within ${hrs(H)}. The signal reaches ×${num(D.margin)} the budget only after <em>${hrs(p.tstar / 3600)}</em>.`; }
        else { ans.innerHTML = `Defensible after <em>${hrs(p.tstar / 3600)}</em>. By the end of the run the signal is ${num(p.end)}× the contamination budget.`; }
        sub.textContent = `Smallest true rate this design can defend in ${hrs(H)}: ${num(p.rmin)} mol s⁻¹ cm⁻² (at margin ×${num(D.margin)}).`;
        // time chart
        const W = Math.max(460, CW - 340), Hh = 300, L = 56, B = 34, T0 = 14, Rr = 16, N = 120;
        const ts = Array.from({ length: N }, (_, i) => (i / (N - 1)) * p.Ts);
        const sig = ts.map((t) => p.s * t), con = ts.map((t) => p.g * t + p.b);
        const raw = Math.max(sig[N - 1], D.margin * con[N - 1]) * 1.05 || 1;
        const step = (() => { const r = raw / 4, p10 = 10 ** Math.floor(Math.log10(r)); return [1, 2, 2.5, 5, 10].map((k) => k * p10).find((k) => k >= r); })();
        const ymax = Math.ceil(raw / step) * step;
        const X = (t) => L + (t / p.Ts) * (W - L - Rr), Y = (v) => Hh - B - (v / ymax) * (Hh - B - T0);
        const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${X(ts[i]).toFixed(1)} ${Y(v).toFixed(1)}`).join("");
        let g = "";
        for (let v = 0; v <= ymax + 1e-9; v += step) { g += `<line x1="${L}" x2="${W - Rr}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e1dacb"/><text class="tick" x="${L - 8}" y="${Y(v) + 3}" text-anchor="end">${num(v, 2)}</text>`; }
        for (let i = 0; i <= 4; i++) { const t = (p.Ts * i) / 4; g += `<text class="tick" x="${X(t)}" y="${Hh - B + 16}" text-anchor="middle">${hrs(t / 3600)}</text>`; }
        g += `<path d="${path(con)}L${X(p.Ts)} ${Y(0)}L${X(0)} ${Y(0)}Z" fill="#c8401f" fill-opacity=".09"/>`;
        g += `<path d="${path(con)}" fill="none" stroke="#c8401f" stroke-width="1.8"/>`;
        g += `<path d="${path(con.map((v) => v * D.margin))}" fill="none" stroke="#c8401f" stroke-width="1" stroke-dasharray="5 4"/>`;
        g += `<path d="${path(sig)}" fill="none" stroke="#1d6b4f" stroke-width="2.6"/>`;
        g += `<text x="${W - Rr}" y="${Y(sig[N - 1]) - 8}" text-anchor="end" class="axl" style="fill:#1d6b4f">true signal</text>`;
        g += `<text x="${W - Rr}" y="${Y(D.margin * con[N - 1]) - 8}" text-anchor="end" class="axl" style="fill:#c8401f">×${num(D.margin)} contamination budget</text>`;
        if (p.tstar !== null && p.tstar <= p.Ts) g += `<line x1="${X(p.tstar)}" x2="${X(p.tstar)}" y1="${T0}" y2="${Hh - B}" stroke="#17150f" stroke-dasharray="3 3"/><text class="axl" x="${X(p.tstar) + 6}" y="${T0 + 10}">t* = ${hrs(p.tstar / 3600)}</text>`;
        g += `<line x1="${L}" x2="${W - Rr}" y1="${Hh - B}" y2="${Hh - B}" stroke="#17150f"/><text class="axl" x="${L}" y="${T0 - 2}">nmol N</text>`;
        m.querySelector(".c1").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}">${g}</svg>`;
        // design map
        const ctx = cv.getContext("2d"), cw = cv.width, ch = cv.height, img = ctx.createImageData(cw, ch);
        const val = [];
        for (let j = 0; j < ch; j++) for (let i = 0; i < cw; i++) {
          const fx = i / (cw - 1), fy = 1 - j / (ch - 1);
          const o = D.mode === "flow" ? { flow: fx * 2.7, ppb: -1 + fy * 5 } : { fixedL: 0.05 + fx * 4.95, ppb: -1 + fy * 5 };
          val.push(Math.log10(plan(D, o).end / D.margin));
        }
        for (let j = 0; j < ch; j++) for (let i = 0; i < cw; i++) {
          const v = val[j * cw + i], k = (j * cw + i) * 4, a = Math.min(1, Math.abs(v) / 2.5) * 0.62 + 0.06;
          const c = v >= 0 ? [29, 107, 79] : [200, 64, 31];
          const col = c.map((x, n) => Math.round([242, 239, 231][n] * (1 - a) + x * a));
          img.data.set([...col, 255], k);
        }
        ctx.putImageData(img, 0, 0);
        const px = D.mode === "flow" ? D.flow / 2.7 : (D.fixedL - 0.05) / 4.95, py = 1 - (D.ppb + 1) / 5;
        m.querySelector(".maplab").textContent = D.mode === "flow" ? "Design space · gas flow × feed purity (log–log)" : "Design space · fixed gas volume × feed purity";
        const xt = D.mode === "flow" ? [1, 10, 100, 500].map((v) => [Math.log10(v) / 2.7, `${v} sccm`]) : [0.05, 1, 2, 3, 4, 5].map((v) => [(v - 0.05) / 4.95, `${v} L`]);
        ov.setAttribute("viewBox", "0 0 100 100"); ov.setAttribute("preserveAspectRatio", "none");
        const at = (fx, fy) => Math.log10(plan(D, D.mode === "flow" ? { flow: fx * 2.7, ppb: -1 + fy * 5 } : { fixedL: 0.05 + fx * 4.95, ppb: -1 + fy * 5 }).end / D.margin);
        const pts = [];
        for (let i = 0; i <= 80; i++) {
          const fx = i / 80; let a0 = 0, b0 = 1;
          if (at(fx, 0) < 0 || at(fx, 1) > 0) continue;
          for (let k = 0; k < 30; k++) { const mid = (a0 + b0) / 2; if (at(fx, mid) >= 0) a0 = mid; else b0 = mid; }
          pts.push(`${(fx * 100).toFixed(2)},${((1 - a0) * 100).toFixed(2)}`);
        }
        ov.innerHTML = (pts.length > 1 ? `<polyline points="${pts.join(" ")}" fill="none" stroke="#17150f" stroke-width="2" vector-effect="non-scaling-stroke"/>` : "") + `<line x1="${px * 100}" x2="${px * 100}" y1="0" y2="100" stroke="#17150f" stroke-width=".25" vector-effect="non-scaling-stroke"/><line y1="${py * 100}" y2="${py * 100}" x1="0" x2="100" stroke="#17150f" stroke-width=".25" vector-effect="non-scaling-stroke"/>`;
        const lab = m.querySelector(".nt-map");
        lab.querySelectorAll(".lbl").forEach((n) => n.remove());
        const put = (html, css) => { const d = el(`<div class="lbl mono" style="position:absolute;font-size:10px;color:#4b473d;${css}">${html}</div>`); lab.appendChild(d); };
        xt.forEach(([f, t]) => put(t, `left:${f * 100}%;bottom:4px;transform:translateX(${f > 0.9 ? -100 : 4}px)`));
        [0.1, 10, 1000].forEach((v) => put(`${v} ppb`, `left:4px;top:${(1 - (Math.log10(v) + 1) / 5) * 100}%;transform:translateY(-50%)`));
        put(`● you`, `left:${px * 100}%;top:${py * 100}%;transform:translate(-4px,-50%);color:#17150f;font-weight:600;font-size:12px`);
      }
      update();
      return m;
    }

    /* --------------------------------------------------------------- stress ---- */
    const ST = { lod: 300, ref: "0 °C", rows: [
      { claim: "Example A", rate: 1e-11, unit: "mol s⁻¹ cm⁻²", area: 1, load: 0, flow: 20 },
      { claim: "Example B", rate: 5, unit: "µg(NH3) h⁻¹ mg_cat⁻¹", area: 1, load: 0.5, flow: 30 },
      { claim: "Example C", rate: 1e-10, unit: "mol s⁻¹ cm⁻²", area: 2, load: 0, flow: 10 },
      { claim: "Example D", rate: 0.5, unit: "µg(NH3) h⁻¹ cm⁻²", area: 1, load: 0, flow: 50 },
    ] };
    const UNITS = { "mol s⁻¹ cm⁻²": (v, a) => v * a, "nmol s⁻¹ cm⁻²": (v, a) => v * 1e-9 * a,
      "µg(NH3) h⁻¹ cm⁻²": (v, a) => (v * 1e-6) / 17.031 / 3600 * a, "µg(NH3) h⁻¹ mg_cat⁻¹": (v, a, l) => (v * 1e-6) / 17.031 / 3600 * l * a };
    function stressPage() {
      const m = el(`<div class="nt-main nt-fade"><div class="nt-kicker"><span class="caps">claim stress-test · continuously purged cells</span></div>
        <h1 class="nt-answer"></h1><p class="nt-head">A reported rate becomes a single number: the reactive-nitrogen level in the feed gas that would supply <i>all</i> of it (η = 1). If that level is below what the gas analysis could bound, the claim can't be separated from feed-gas contamination, whatever else was done. The rows are illustrative inputs; type in any paper's numbers.</p>
        <section class="nt-sec"><header><h3>Claims</h3><span class="caps"><span class="refseg"></span></span></header><div class="tbl"></div>
          <button class="nt-btn ghost add" style="margin-top:10px">+ Add claim</button></section>
        <section class="nt-sec"><header><h3>Equivalent feed-gas contamination</h3><span class="caps">ppb N · log scale</span></header>
          <div class="nt-ctl" style="max-width:420px"><div class="row"><label>Your gas analysis can bound total reactive N down to</label><span class="val lodv"></span></div><input class="lod" type="range" min="-2" max="3" step="0.05"></div>
          <div class="nt-chart plot"></div></section></div>`);
      const tbl = m.querySelector(".tbl"), lod = m.querySelector(".lod");
      lod.value = Math.log10(ST.lod);
      lod.oninput = () => { ST.lod = 10 ** +lod.value; draw(); };
      const refseg = el(`<div class="nt-seg" style="margin:0">${Object.keys(REF).map((k) => `<button data-v="${k}">${k} ref.</button>`).join("")}</div>`);
      m.querySelector(".refseg").appendChild(refseg);
      refseg.querySelectorAll("button").forEach((b) => b.onclick = () => { ST.ref = b.dataset.v; draw(); });
      m.querySelector(".add").onclick = () => { ST.rows.push({ claim: `Claim ${ST.rows.length + 1}`, rate: 1e-11, unit: "mol s⁻¹ cm⁻²", area: 1, load: 0, flow: 20 }); table(); draw(); };
      const eq = (r) => { const T = REF[ST.ref]; const ndot = 101325 * r.flow * 1e-6 / 60 / (R * T); return (UNITS[r.unit](+r.rate, +r.area, +r.load) / ndot) * 1e9; };
      function table() {
        tbl.innerHTML = `<table class="nt-tbl nt-stbl"><tr><th>Claim</th><th>Rate</th><th>Unit</th><th>Area cm²</th><th>Loading mg cm⁻²</th><th>Flow sccm</th><th style="text-align:right">≡ feed ppb</th><th></th></tr>
          ${ST.rows.map((r, i) => `<tr data-i="${i}"><td><input data-k="claim" value="${esc(r.claim)}"></td><td><input data-k="rate" value="${r.rate}"></td>
            <td><select data-k="unit">${Object.keys(UNITS).map((u) => `<option ${u === r.unit ? "selected" : ""}>${u}</option>`).join("")}</select></td>
            <td><input data-k="area" value="${r.area}"></td><td><input data-k="load" value="${r.load}"></td><td><input data-k="flow" value="${r.flow}"></td>
            <td class="n eq"></td><td><button class="nt-x" title="remove">×</button></td></tr>`).join("")}</table>`;
        tbl.querySelectorAll("tr[data-i]").forEach((tr) => {
          const i = +tr.dataset.i;
          tr.querySelectorAll("[data-k]").forEach((inp) => inp.addEventListener("input", () => { ST.rows[i][inp.dataset.k] = inp.dataset.k === "claim" || inp.dataset.k === "unit" ? inp.value : +inp.value; draw(); }));
          tr.querySelector(".nt-x").onclick = () => { ST.rows.splice(i, 1); table(); draw(); };
        });
      }
      function draw() {
        refseg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === ST.ref));
        m.querySelector(".lodv").textContent = `${num(ST.lod)} ppb`;
        const vals = ST.rows.map(eq);
        tbl.querySelectorAll("tr[data-i]").forEach((tr, i) => { const v = vals[i]; tr.querySelector(".eq").innerHTML = `<span style="color:${v > ST.lod ? "#1d6b4f" : "#c8401f"}">${num(v)}</span>`; });
        const bad = vals.filter((v) => !(v > ST.lod)).length;
        m.querySelector(".nt-answer").innerHTML = bad ? `<em>${bad} of ${vals.length}</em> claims sit below your gas-analysis bound: even a clean measurement at that sensitivity couldn't rule out feed gas as their sole source.` : `All ${vals.length} claims sit above your gas-analysis bound. A clean measurement at that sensitivity could rule out feed gas as their sole source; other sources still need bounding.`;
        const W = Math.max(520, CW), L = 150, Rr = 20, RH = 34, H = 40 + ST.rows.length * RH + 30, lo = 1e-3, hi = 1e6;
        const X = (v) => L + (Math.log10(Math.min(Math.max(v, lo), hi)) + 3) / 9 * (W - L - Rr);
        let g = `<rect x="${L}" y="20" width="${X(ST.lod) - L}" height="${H - 50}" fill="#c8401f" fill-opacity=".07"/>`;
        for (let e = -3; e <= 6; e++) g += `<line x1="${X(10 ** e)}" x2="${X(10 ** e)}" y1="20" y2="${H - 30}" stroke="#e1dacb"/><text class="tick" x="${X(10 ** e)}" y="${H - 12}" text-anchor="middle">${tickLabel(10 ** e)}</text>`;
        g += `<line x1="${X(ST.lod)}" x2="${X(ST.lod)}" y1="12" y2="${H - 30}" stroke="#c8401f" stroke-width="1.5"/><text class="axl" x="${X(ST.lod) + 6}" y="16" style="fill:#c8401f">your detection bound</text>`;
        ST.rows.forEach((r, i) => {
          const y = 40 + i * RH, v = vals[i], ok = v > ST.lod;
          g += `<text class="axl" x="0" y="${y + 4}">${esc(r.claim)}</text><line x1="${L}" x2="${X(v)}" y1="${y}" y2="${y}" stroke="#c6bda9" stroke-dasharray="2 3"/>
            <circle cx="${X(v)}" cy="${y}" r="6" fill="${ok ? "#1d6b4f" : "#c8401f"}"/><text class="tick" x="${X(v) + 10}" y="${y + 3}">${num(v)} ppb</text>`;
        });
        m.querySelector(".plot").innerHTML = `<svg viewBox="0 0 ${W} ${H}">${g}</svg>`;
      }
      table(); draw();
      return m;
    }

    /* ------------------------------------------------------------- evidence ---- */
    function evidencePage(model) {
      const res = model.get("agent_res") || {};
      const m = el(`<div class="nt-main nt-fade"><div class="nt-kicker"><span class="caps">the agent proposes · the scientist confirms</span></div>
        <h1 class="nt-answer">Paste a paper's methods. Get evidence with <em>verbatim quotes</em>, or nothing.</h1>
        <p class="nt-head">A deterministic pass finds passages about product amounts, gas flow, duration, blanks, impurities and isotopes. With a key, Claude proposes structured candidates; any candidate whose quote isn't found verbatim in your text is discarded. Nothing enters an audit until you type it into the record.</p>
        <section class="nt-sec"><header><h3>Source text</h3><span class="caps">${esc(res.model || "claude-opus-5-5")}</span></header>
          <textarea class="nt-field txt" rows="12" placeholder="Paste the experimental section and results…"></textarea>
          <div style="display:grid;grid-template-columns:1fr auto;gap:10px;margin-top:10px;align-items:center">
            <input class="nt-field key" type="password" placeholder="Anthropic API key (optional; stays in this kernel)">
            <div><button class="nt-btn ghost find">Find passages</button><button class="nt-btn ask">Ask Claude</button></div></div></section>
        <div class="out"></div></div>`);
      const txt = m.querySelector(".txt"), key = m.querySelector(".key");
      txt.value = model.get("agent_text") || "";
      txt.oninput = () => { model.set("agent_text", txt.value); model.save_changes(); };
      const go = (llm) => { model.set("agent_req", { text: txt.value, key: key.value, llm, n: Date.now() }); model.save_changes(); m.querySelectorAll(".nt-btn").forEach((b) => (b.disabled = true)); m.querySelector(".out").innerHTML = `<p class="nt-empty" style="margin-top:20px">${llm ? "Claude is reading…" : "Searching…"}</p>`; };
      m.querySelector(".find").onclick = () => go(false);
      m.querySelector(".ask").onclick = () => go(true);
      const out = m.querySelector(".out");
      if (res.error) out.appendChild(el(`<div class="nt-err" style="margin-top:20px">${esc(res.error)}</div>`));
      if (res.candidates) out.appendChild(el(`<section class="nt-sec"><header><h3>Candidates</h3><span class="caps">${res.candidates.length} verified · ${res.dropped} dropped</span></header>
        <table class="nt-tbl"><tr><th>Field</th><th style="text-align:right">Value</th><th>Unit</th><th>Species</th><th>Quote</th></tr>${res.candidates.map((c) => `<tr><td>${esc(c.field)}</td><td class="n">${c.value ?? "—"}</td><td>${esc(c.unit ?? "")}</td><td>${esc(c.species ?? "")}</td><td><span class="serif" style="font-style:italic;font-size:14px">“${esc(c.quote)}”</span></td></tr>`).join("")}</table></section>`));
      if (res.passages) out.appendChild(el(`<section class="nt-sec"><header><h3>Passages</h3><span class="caps">${res.passages.length} found</span></header>
        ${res.passages.map((p) => `<div class="nt-proto"><div class="k">§${p.sentence}</div><div><span class="caps">${esc(p.topics)}</span><div class="quote" style="font-size:15px">${esc(p.text)}</div>${p.regex_values ? `<div class="cite">${esc(p.regex_values)}</div>` : ""}</div></div>`).join("") || '<p class="nt-empty">No relevant passages.</p>'}</section>`));
      return m;
    }

    /* --------------------------------------------------------------- report ---- */
    function reportPage(model, a) {
      const m = el(`<div class="nt-main nt-fade"></div>`);
      if (!a || !a.report) { m.innerHTML = `<p class="nt-empty">${esc(a && a.error ? a.error : "No report yet.")}</p>`; return m; }
      const bar = el(`<div style="margin-bottom:22px;display:flex;flex-wrap:wrap;gap:8px"><button class="nt-btn">Case file (.md)</button><button class="nt-btn ghost">Independent-review pack (.md)</button><button class="nt-btn ghost">Audit report (.md)</button><button class="nt-btn ghost">JSON</button><button class="nt-btn ghost">CSV</button></div>`);
      const [b1, b2, b3, b4, b5] = bar.querySelectorAll("button");
      b1.onclick = () => download(`${a.case_id}_casefile.md`, a.report.casefile, "text/markdown");
      b2.onclick = () => download(`${a.case_id}_review_pack.md`, a.report.review, "text/markdown");
      b3.onclick = () => download(`${a.case_id}_audit.md`, a.report.md, "text/markdown");
      b4.onclick = () => download(`${a.case_id}_audit.json`, a.report.json, "application/json");
      b5.onclick = () => download(`${a.case_id}_contributions.csv`, a.report.csv, "text/csv");
      m.appendChild(bar);
      m.appendChild(el(`<p class="nt-head" style="font-size:13px;margin:0 0 14px">The case file is the complete, source-linked account: values as reported with their quotes, what was counted or set aside and why, the calculation, the qualified conclusion, what would close the gap, and the review status. The review pack is a checklist for an independent domain researcher.</p>`));
      m.appendChild(el(`<article class="nt-doc">${md(a.report.casefile)}</article>`));
      return m;
    }

    /* --------------------------------------------------------------- method ---- */
    function methodPage(model) {
      const b = model.get("bench") || {};
      const props = Object.entries(b.properties || {});
      const m = el(`<div class="nt-main nt-fade"><div class="nt-kicker"><span class="caps">method · evidence · limits</span></div>
        <h1 class="nt-verdict">A signal above <em>unmeasured</em> contamination is not yet a result.</h1>
        <div class="nt-essay"><p>Electrochemical N₂-to-NH₃ studies report nanomoles of ammonia, and ammonia and NOx in gases, electrolytes, membranes and ionomers produce the same signal. Protocols say which controls to run. NitroTrace computes whether the controls that were actually run exclude contamination. When they don't, it works out which measurement, at what level, would.</p>
        <p>Every input carries species, basis, unit, bound type, matching status and a quoted source. Missing evidence is <i>unknown</i>, never zero. Supplier purity, upstream-of-purifier gas and non-matched blanks are shown but never counted. Overlapping blanks need an explicit coverage model. Bounds are combined as intervals, nothing is fitted, and the verdict follows five ordered rules.</p></div>
        <section class="nt-sec"><header><h3>Benchmark</h3><span class="caps">${b.n_synthetic} synthetic records with independent expected values</span></header>
          <div class="nt-kpis"><div><b class="nt-big">${b.nitrotrace_correct}/${b.n_synthetic}</b><span>NitroTrace verdicts correct</span></div>
          <div><b class="nt-big" style="color:#8a8474">${b.naive_correct}/${b.n_synthetic}</b><span>naive "sum every number, unknown = 0"</span></div>
          <div><b class="nt-big" style="color:#8a8474">${b.checklist_correct}/${b.n_synthetic}</b><span>protocol checklist, no arithmetic</span></div>
          <div><b class="nt-big" style="color:#c8401f">${b.false_exceeds_nitrotrace}<span style="font-size:.4em;color:#8a8474;display:inline"> vs ${b.false_exceeds_naive} vs ${b.false_exceeds_checklist}</span></b><span>false "exceeds" claims (NitroTrace / naive / checklist)</span></div></div>
          <p style="font-size:13px;color:var(--ink-2)">Max relative error against ${b.n_numeric} hand-computed references: <span class="mono">${num(b.max_rel_error)}</span>. Reproduces Choi et al. 2020 Table 1 within 0.05 %. The whole suite, including the property checks below, runs live in this kernel in <span class="mono">${num(b.elapsed_ms)} ms</span>.</p></section>
        <section class="nt-sec"><header><h3>Machine-checked properties</h3><span class="caps">run on every record, every launch</span></header>
          ${props.map(([k, [p, n, f]]) => `<div class="nt-prop"><span class="${p === n ? "ok" : "bad"}"></span><span>${esc(k)}</span><span class="mono">${p}/${n}</span></div>`).join("")}
          <p style="font-size:12px;color:var(--ink-3)">Scope: unit invariance skips records with input errors. "Unknown is never zero" skips measurements set aside as overlaps, because deleting one of two conflicting measurements removes the conflict itself. Monotonicity skips records with a whole-system blank.</p></section>
        <section class="nt-sec"><header><h3>Review status</h3><span class="caps">stated plainly</span></header>
          <p style="font-size:14.5px;color:var(--ink-2);max-width:820px">The published records have been checked against their sources by the NitroTrace authors only. They have <b>not yet been independently reviewed</b> by a domain researcher. Every case file says so, and the Report page exports an independent-review checklist for each record.</p></section>
        <section class="nt-sec"><header><h3>The trap, on real data</h3><span class="caps">Bragulla et al. 2026 · open circuit</span></header>
          <p style="font-size:14.5px;color:var(--ink-2);max-width:820px">Ammonium measured with <i>no electrochemistry at all</i> exceeds every separately quantified material release. The naive sum and the checklist both call it "exceeds". NitroTrace calls it "insufficient", because the in-cell component release was never measured under matched conditions. The authors trace the difference to the Nafion ionomer.</p></section>
        <section class="nt-sec"><header><h3>Limits</h3><span class="caps">stated, not hidden</span></header><ol class="nt-list">
          <li>Aqueous electrochemical N₂→NH₃ only; independent interval arithmetic; no kinetics.</li>
          <li>η = 1 is a labelled conservative scenario, not a fitted efficiency.</li>
          <li>Synthetic records test software and reasoning, not attribution in real laboratories. Published records carry no ground truth and are not scored.</li>
          <li>The control rulebook has not yet been reviewed by an independent chemist. Outputs describe what the entered evidence permits, not the original studies.</li></ol></section>
        <section class="nt-sec"><header><h3>Sources</h3><span class="caps"></span></header><ol class="nt-list">
          <li>[S1] Andersen et al., <i>Nature</i> 570, 504–508 (2019).</li><li>[S2] Choi et al., <i>Nat. Commun.</i> 11, 5546 (2020).</li>
          <li>[S3] Bragulla et al., <i>ChemSusChem</i> 19, e202501974 (2026).</li><li>[S4] Hu et al., <i>Natl. Sci. Rev.</i> (2025).</li></ol></section></div>`);
      return m;
    }

    /* ----------------------------------------------------------------- shell ---- */
    function render({ model, el: host }) {
      if (!document.getElementById("nt-fonts")) {
        const l = document.createElement("link"); l.id = "nt-fonts"; l.rel = "stylesheet";
        l.href = "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap";
        document.head.appendChild(l);
      }
      const b = model.get("bench") || {};
      const propOk = Object.values(b.properties || {}).reduce((s, [p]) => s + p, 0), propN = Object.values(b.properties || {}).reduce((s, [, n]) => s + n, 0);
      const root = el(`<div class="nt">
        <div class="nt-mast"><div class="nt-mark">Nitro<i>Trace</i></div>
          <div class="nt-tag">Given an ammonia measurement and the controls actually run: <b>what contamination remains unbounded, and how stringent must new measurements be to close the gap?</b></div>
          <div class="nt-figs"><div class="nt-fig"><b>${b.nitrotrace_correct}/${b.n_synthetic}</b><span>verdicts</span></div><div class="nt-fig"><b>${propOk}/${propN}</b><span>proofs</span></div><div class="nt-fig"><b>0.05%</b><span>vs Choi 2020</span></div></div></div>
        <nav class="nt-nav"></nav><div class="nt-body"></div></div>`);
      host.appendChild(root);
      const nav = root.querySelector(".nt-nav"), body = root.querySelector(".nt-body");
      TABS.forEach(([k, lab], i) => { const bt = el(`<button data-k="${k}"><span class="n">0${i + 1}</span>${lab}</button>`); bt.onclick = () => { model.set("tab", k); model.save_changes(); draw(); }; nav.appendChild(bt); });
      let q = "";
      function index() {
        const cat = model.get("catalog") || [], sel = model.get("selected");
        const box = el(`<aside class="nt-index"><input placeholder="Search records…"></aside>`);
        const inp = box.querySelector("input"); inp.value = q;
        const list = el(`<div></div>`); box.appendChild(list);
        const fill = () => {
          list.innerHTML = "";
          for (const [grp, name] of [["published", "Published records"], ["synthetic", "Synthetic benchmark"]]) {
            const items = cat.filter((c) => c.group === grp && (c.title + c.id).toLowerCase().includes(q.toLowerCase()));
            if (!items.length) continue;
            list.appendChild(el(`<h6 class="caps">${name} · ${items.length}</h6>`));
            for (const c of items) {
              const r = el(`<div class="nt-rec ${c.key === sel ? "on" : ""}" title="${esc(c.status_label)}"><div class="sq" style="background:${S[c.status].hex}"></div><div><div class="t">${esc(c.title)}</div><div class="s">${esc(c.sub)}</div></div></div>`);
              r.onclick = () => { model.set("selected", c.key); model.set("scale", 1); model.save_changes(); };
              list.appendChild(r);
            }
          }
        };
        inp.oninput = () => { q = inp.value; fill(); };
        fill(); return box;
      }
      function draw() {
        const tab = model.get("tab") || "audit", a = model.get("audit");
        nav.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset.k === tab));
        const withIndex = ["audit", "settle", "report"].includes(tab);
        body.className = "nt-body" + (withIndex ? "" : " solo");
        const scroll = body.querySelector(".nt-index")?.scrollTop || 0;
        const full = root.clientWidth || 1000;
        CW = Math.round((withIndex && full > 900 ? full - 250 : full) - 70);
        body.innerHTML = "";
        if (withIndex) { const ix = index(); body.appendChild(ix); ix.scrollTop = scroll; }
        const page = { audit: () => auditPage(model, a), settle: () => settlePage(model, a), design: () => designPage(model),
          stress: () => stressPage(), evidence: () => evidencePage(model), report: () => reportPage(model, a), method: () => methodPage(model) }[tab];
        body.appendChild(page());
      }
      model.on("change:audit", () => { if (["audit", "settle", "report"].includes(model.get("tab") || "audit")) draw(); });
      model.on("change:tab", draw);
      model.on("change:agent_res", () => { if (model.get("tab") === "evidence") draw(); });
      let lastW = 0;
      new ResizeObserver(() => { const w = root.clientWidth; if (w && Math.abs(w - lastW) > 40) { lastW = w; draw(); } }).observe(root);
      draw();
    }
    export default { render };
    '''

    NT_CSS2 = r'''.nt {
      --paper: #f2efe7; --paper-2: #e9e4d8; --card: #faf8f3; --ink: #17150f; --ink-2: #4b473d; --ink-3: #8a8474;
      --rule: #d8d1c1; --rule-2: #c6bda9;
      --red: #c8401f; --green: #1d6b4f; --ochre: #a86e12; --violet: #4f46a5; --blue: #20507a;
      --serif: "Instrument Serif", "Iowan Old Style", Georgia, serif;
      --sans: "Inter Tight", "Inter", system-ui, sans-serif;
      --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", monospace;
      width: 100%; container-type: inline-size; container-name: nt;
      background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 14px; line-height: 1.5;
      border: 1px solid var(--rule); border-radius: 4px; overflow: hidden;
      font-feature-settings: "ss01", "cv11";
    }
    .nt * { box-sizing: border-box; }
    .nt button, .nt input, .nt textarea, .nt select { font: inherit; color: inherit; }
    .nt .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
    .nt .serif { font-family: var(--serif); font-weight: 400; }
    .nt .caps { font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase; font-weight: 600; color: var(--ink-3); }

    /* masthead */
    .nt-mast { display: grid; grid-template-columns: auto 1fr auto; align-items: end; gap: 28px;
      padding: 22px 32px 14px; border-bottom: 1px solid var(--ink); }
    .nt-mark { font-family: var(--serif); font-size: 40px; line-height: .9; letter-spacing: -.01em; }
    .nt-mark i { font-style: italic; }
    .nt-mark sub { font-size: .45em; vertical-align: baseline; position: relative; top: .05em; }
    .nt-tag { color: var(--ink-2); max-width: 520px; font-size: 13px; padding-bottom: 3px; }
    .nt-figs { display: flex; gap: 22px; padding-bottom: 3px; }
    .nt-fig { text-align: right; }
    .nt-fig b { display: block; font-family: var(--mono); font-size: 17px; font-weight: 500; }
    .nt-fig span { font-size: 10.5px; color: var(--ink-3); letter-spacing: .06em; text-transform: uppercase; }
    .nt-nav { display: flex; gap: 0; padding: 0 32px; border-bottom: 1px solid var(--rule); background: var(--paper); }
    .nt-nav button { background: none; border: 0; padding: 12px 0 11px; margin-right: 26px; cursor: pointer;
      color: var(--ink-3); font-weight: 500; border-bottom: 2px solid transparent; transition: color .15s; }
    .nt-nav button:hover { color: var(--ink); }
    .nt-nav button.on { color: var(--ink); border-bottom-color: var(--ink); }
    .nt-nav button .n { font-family: var(--mono); font-size: 11px; margin-right: 6px; color: var(--ink-3); }

    /* layout */
    .nt-body { display: grid; grid-template-columns: 250px minmax(0, 1fr); min-height: 760px; }
    .nt-body.solo { grid-template-columns: 1fr; }
    .nt-index { border-right: 1px solid var(--rule); background: var(--paper-2); padding: 16px 0; max-height: 1400px; overflow: auto; }
    .nt-index input { width: calc(100% - 32px); margin: 0 16px 12px; padding: 7px 10px; border: 1px solid var(--rule-2);
      background: var(--card); border-radius: 3px; outline: none; }
    .nt-index input:focus { border-color: var(--ink); }
    .nt-index h6 { margin: 14px 16px 6px; }
    .nt-rec { display: grid; grid-template-columns: 10px 1fr; gap: 10px; padding: 7px 16px; cursor: pointer; align-items: start;
      border-left: 3px solid transparent; }
    .nt-rec:hover { background: #e2dccd; }
    .nt-rec.on { background: var(--card); border-left-color: var(--ink); }
    .nt-rec .sq { width: 9px; height: 9px; margin-top: 5px; }
    .nt-rec .t { font-size: 12.5px; line-height: 1.35; color: var(--ink); }
    .nt-rec .s { font-size: 11px; color: var(--ink-3); }
    .nt-main { padding: 28px 34px 48px; min-width: 0; }

    /* verdict */
    .nt-kicker { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
    .nt-pill { display: inline-flex; align-items: center; gap: 7px; padding: 2px 9px 2px 7px; border: 1px solid currentColor;
      border-radius: 2px; font-size: 10.5px; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; }
    .nt-pill i { width: 8px; height: 8px; background: currentColor; display: block; }
    .nt-verdict { font-family: var(--serif); font-size: clamp(32px, 4.2cqi, 50px); line-height: 1.04; letter-spacing: -.012em; margin: 4px 0 12px; max-width: 980px; }
    .nt-verdict em { font-style: italic; }
    .nt-head { color: var(--ink-2); max-width: 860px; font-size: 15px; }
    .nt-title { font-size: 13px; color: var(--ink-2); }

    /* sections */
    .nt-sec { margin-top: 34px; }
    .nt-sec > header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px;
      border-top: 1px solid var(--ink); padding-top: 8px; margin-bottom: 14px; }
    .nt-sec > header h3 { margin: 0; font-family: var(--serif); font-weight: 400; font-size: 24px; }
    .nt-sec > header .caps { white-space: nowrap; }
    .nt-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
    .nt-cols3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; border: 1px solid var(--rule); }
    .nt-cov { padding: 14px 16px; border-right: 1px solid var(--rule); background: var(--card); }
    .nt-cov:last-child { border-right: 0; }
    .nt-cov .v { font-family: var(--serif); font-size: 22px; margin-top: 2px; }
    .nt-cov .bar { height: 3px; margin: 10px 0 8px; }
    .nt-cov p { margin: 0; font-size: 12.5px; color: var(--ink-2); }
    .nt-list { margin: 0; padding: 0; list-style: none; counter-reset: li; }
    .nt-list li { position: relative; padding: 8px 0 8px 30px; border-bottom: 1px dotted var(--rule-2); font-size: 13.5px; }
    .nt-list li::before { counter-increment: li; content: counter(li, decimal-leading-zero); position: absolute; left: 0; top: 9px;
      font-family: var(--mono); font-size: 11px; color: var(--ink-3); }

    /* ruler */
    .nt-ruler svg { width: 100%; display: block; overflow: visible; }
    .nt-ruler .lab { font-size: 12px; fill: var(--ink); }
    .nt-ruler .sub { font-size: 10.5px; fill: var(--ink-3); font-family: var(--mono); }
    .nt-ruler .tick { font-family: var(--mono); font-size: 10px; fill: var(--ink-3); }
    .nt-grow { transform-box: fill-box; transform-origin: left center; animation: nt-grow .7s cubic-bezier(.2,.8,.2,1) both; }
    @keyframes nt-grow { from { transform: scaleX(0); } }
    .nt-fade { animation: nt-fade .5s ease both; }
    @keyframes nt-fade { from { opacity: 0; transform: translateY(4px); } }

    /* scale track */
    .nt-track { position: relative; height: 54px; margin: 6px 0 0; user-select: none; touch-action: none; cursor: ew-resize; }
    .nt-track .seg { position: absolute; top: 14px; height: 14px; }
    .nt-track .thumb { position: absolute; top: 4px; width: 2px; height: 34px; background: var(--ink); transform: translateX(-1px); }
    .nt-track .thumb::after { content: attr(data-v); position: absolute; top: -2px; left: 8px; font-family: var(--mono);
      font-size: 11px; background: var(--ink); color: var(--paper); padding: 1px 6px; white-space: nowrap; }
    .nt-track .ax { position: absolute; top: 33px; font-family: var(--mono); font-size: 10px; color: var(--ink-3); transform: translateX(-50%); }

    /* table */
    .nt-tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    .nt-tbl th { text-align: left; font-weight: 600; font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3);
      border-bottom: 1px solid var(--ink); padding: 6px 10px 6px 0; }
    .nt-tbl td { padding: 7px 10px 7px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
    .nt-tbl td.n { font-family: var(--mono); text-align: right; white-space: nowrap; }
    .nt-tbl tr.off td { color: var(--ink-3); }
    .nt-role { font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; font-weight: 600; white-space: nowrap; }

    /* details */
    .nt details { border-top: 1px solid var(--rule); }
    .nt details:last-child { border-bottom: 1px solid var(--rule); }
    .nt summary { cursor: pointer; padding: 11px 0; list-style: none; display: flex; justify-content: space-between; font-weight: 500; }
    .nt summary::-webkit-details-marker { display: none; }
    .nt summary::after { content: "+"; font-family: var(--mono); color: var(--ink-3); }
    .nt details[open] summary::after { content: "–"; }
    .nt .trail { font-family: var(--mono); font-size: 11.5px; color: var(--ink-2); padding: 0 0 14px; }
    .nt .trail div { padding: 2px 0; }
    .nt .trail b { font-family: var(--sans); font-size: 12.5px; color: var(--ink); display: block; margin-top: 10px; }
    .nt .quote { font-family: var(--serif); font-style: italic; font-size: 16px; line-height: 1.35; color: var(--ink);
      border-left: 2px solid var(--ink); padding: 2px 0 2px 14px; margin: 8px 0; }
    .nt .cite { font-size: 11px; color: var(--ink-3); font-family: var(--mono); }
    .nt-proto { padding: 16px 0; border-bottom: 1px solid var(--rule); display: grid; grid-template-columns: 54px 1fr; gap: 14px; }
    .nt-proto .k { font-family: var(--mono); font-size: 11px; color: var(--ink-3); padding-top: 4px; }
    .nt-proto h4 { margin: 0 0 4px; font-size: 15px; font-weight: 600; }
    .nt-proto p { margin: 2px 0; color: var(--ink-2); font-size: 13px; }

    /* settle */
    .nt-spec { display: grid; grid-template-columns: 54px 1fr 280px; gap: 18px; padding: 20px 0; border-bottom: 1px solid var(--rule); align-items: start; }
    .nt-spec .k { font-family: var(--mono); font-size: 11px; color: var(--ink-3); padding-top: 8px; }
    .nt-spec h4 { margin: 0 0 4px; font-size: 16px; }
    .nt-spec p { margin: 0; color: var(--ink-2); font-size: 13px; max-width: 620px; }
    .nt-spec .big { text-align: right; }
    .nt-spec .big b { font-family: var(--serif); font-size: 44px; line-height: 1; font-weight: 400; display: block; }
    .nt-spec .big span { font-family: var(--mono); font-size: 11px; color: var(--ink-3); }
    .nt-gap { display: grid; grid-template-columns: repeat(3, 1fr); border: 1px solid var(--ink); background: var(--card); }
    .nt-gap > div { padding: 16px 18px; border-right: 1px solid var(--rule); }
    .nt-gap > div:last-child { border-right: 0; }
    .nt-gap b { display: block; font-family: var(--serif); font-size: 36px; font-weight: 400; line-height: 1.1; }
    .nt-gap span { font-size: 12px; color: var(--ink-2); }
    .nt-note { background: var(--card); border: 1px solid var(--rule); border-left: 3px solid var(--ochre); padding: 10px 14px; margin: 10px 0; font-size: 13px; color: var(--ink-2); }

    .nt-as { margin: 8px 0 0; padding: 0 0 0 16px; font-size: 12px; color: var(--ink-3); }
    .nt-as li { margin: 2px 0; }
    .nt-note.caveat { border-left-color: var(--ink); }
    .nt-hyp { border: 1px dashed var(--violet); border-left: 3px solid var(--violet); background: #f3f1fb; color: #2f2a6b;
      padding: 12px 16px; margin: 10px 0; font-size: 13.5px; }
    .nt-ba { display: flex; align-items: center; gap: 10px; margin: 8px 0; flex-wrap: wrap; }
    .nt-ba .arr { font-family: var(--mono); color: var(--ink-3); }
    .nt-wi { display: grid; gap: 8px; }
    .nt-wirow { display: grid; grid-template-columns: 22px 1fr auto 140px 90px; gap: 10px; align-items: center;
      background: var(--card); border: 1px solid var(--rule); padding: 8px 12px; font-size: 13.5px; }
    .nt-wirow.off { opacity: .6; }
    .nt-wirow small { display: block; color: var(--red); font-size: 11.5px; }
    .nt-wirow .u { font-family: var(--mono); font-size: 12px; color: var(--ink-3); }

    /* controls */
    .nt-ctl { margin: 0 0 16px; }
    .nt-ctl .row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
    .nt-ctl label { font-size: 12.5px; color: var(--ink-2); }
    .nt-ctl .val { font-family: var(--mono); font-size: 13px; font-weight: 500; }
    .nt input[type=range] { -webkit-appearance: none; appearance: none; width: 100%; height: 18px; background: transparent; cursor: pointer; }
    .nt input[type=range]::-webkit-slider-runnable-track { height: 2px; background: var(--ink); }
    .nt input[type=range]::-moz-range-track { height: 2px; background: var(--ink); }
    .nt input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 18px; margin-top: -8px; background: var(--paper);
      border: 2px solid var(--ink); border-radius: 0; }
    .nt input[type=range]::-moz-range-thumb { width: 10px; height: 16px; background: var(--paper); border: 2px solid var(--ink); border-radius: 0; }
    .nt-seg { display: inline-flex; border: 1px solid var(--ink); margin-bottom: 18px; }
    .nt-seg button { background: none; border: 0; padding: 6px 14px; cursor: pointer; font-size: 12.5px; }
    .nt-seg button.on { background: var(--ink); color: var(--paper); }
    .nt-design { display: grid; grid-template-columns: 300px 1fr; gap: 40px; }
    .nt-chart svg { width: 100%; display: block; }
    .nt-chart .tick { font-family: var(--mono); font-size: 10px; fill: var(--ink-3); }
    .nt-chart .axl { font-size: 11px; fill: var(--ink-2); }
    .nt-map { position: relative; cursor: crosshair; }
    .nt-map canvas { width: 100%; display: block; image-rendering: auto; }
    .nt-map svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
    .nt-answer { font-family: var(--serif); font-size: 32px; line-height: 1.12; margin: 0 0 6px; max-width: 900px; }

    /* inputs */
    .nt-field { width: 100%; padding: 7px 9px; border: 1px solid var(--rule-2); background: var(--card); border-radius: 2px; outline: none; }
    .nt-field:focus { border-color: var(--ink); }
    .nt textarea.nt-field { font-family: var(--mono); font-size: 12px; line-height: 1.45; resize: vertical; }
    .nt .nt-btn { background: var(--ink); color: var(--paper); border: 1px solid var(--ink); padding: 8px 16px; cursor: pointer; font-weight: 600; border-radius: 2px; }
    .nt .nt-btn.ghost { background: transparent; color: var(--ink); }
    .nt .nt-btn:disabled { opacity: .5; cursor: progress; }
    .nt .nt-btn + .nt-btn { margin-left: 8px; }
    .nt-stbl input, .nt-stbl select { width: 100%; border: 0; border-bottom: 1px solid var(--rule-2); background: transparent; padding: 4px 2px; font-family: var(--mono); font-size: 12px; outline: none; }
    .nt-stbl input:focus, .nt-stbl select:focus { border-bottom-color: var(--ink); }
    .nt-x { background: none; border: 0; color: var(--ink-3); cursor: pointer; font-size: 16px; }

    /* report */
    .nt-doc { max-width: 860px; background: var(--card); border: 1px solid var(--rule); padding: 36px 44px; }
    .nt-doc h1 { font-family: var(--serif); font-weight: 400; font-size: 34px; line-height: 1.1; margin: 0 0 14px; }
    .nt-doc h2 { font-family: var(--serif); font-weight: 400; font-size: 22px; margin: 26px 0 8px; border-top: 1px solid var(--ink); padding-top: 8px; }
    .nt-doc h3 { font-size: 14px; margin: 16px 0 4px; }
    .nt-doc p, .nt-doc li { font-size: 13.5px; color: var(--ink-2); }
    .nt-doc blockquote { font-family: var(--serif); font-style: italic; font-size: 15px; border-left: 2px solid var(--ink); margin: 8px 0; padding-left: 12px; }
    .nt-doc code { font-family: var(--mono); font-size: 12px; background: var(--paper-2); padding: 0 4px; }

    /* method */
    .nt-essay { max-width: 760px; }
    .nt-essay p { font-size: 15.5px; line-height: 1.6; color: var(--ink-2); margin: 0 0 14px; }
    .nt-essay p:first-of-type::first-letter { font-family: var(--serif); float: left; font-size: 64px; line-height: .8; padding: 6px 8px 0 0; color: var(--ink); }
    .nt-big { font-family: var(--serif); font-size: 56px; line-height: 1; }
    .nt-kpis { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink); }
    .nt-kpis > div { padding: 18px 18px 16px 0; border-right: 1px solid var(--rule); }
    .nt-kpis > div + div { padding-left: 18px; }
    .nt-kpis > div:last-child { border-right: 0; }
    .nt-kpis span { display: block; font-size: 12px; color: var(--ink-2); margin-top: 6px; }
    .nt-prop { display: grid; grid-template-columns: 22px 1fr auto; gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--rule); align-items: baseline; }
    .nt-prop .ok { width: 12px; height: 12px; background: var(--green); display: inline-block; }
    .nt-prop .bad { width: 12px; height: 12px; background: var(--red); display: inline-block; }
    .nt-empty { color: var(--ink-3); font-style: italic; }
    .nt-err { border: 1px solid var(--violet); border-left-width: 3px; background: var(--card); padding: 12px 16px; font-family: var(--mono); font-size: 12px; color: var(--violet); }

    @container nt (max-width: 900px) {
      .nt-body { grid-template-columns: 1fr; }
      .nt-index { max-height: 260px; border-right: 0; border-bottom: 1px solid var(--rule); }
      .nt-cols, .nt-design { grid-template-columns: 1fr; }
      .nt-mast { grid-template-columns: 1fr; }
      .nt-figs { justify-content: flex-start; }
      .nt-verdict { font-size: 34px; }
      .nt-spec { grid-template-columns: 40px 1fr; }
      .nt-spec .big { grid-column: 2; text-align: left; }
      .nt-kpis, .nt-gap, .nt-cols3 { grid-template-columns: 1fr 1fr; }
    }
    '''

    return NT_CSS2, NT_ESM


@app.cell(hide_code=True)
def ui_widget(
    CASE_KEYS,
    DATA,
    ExperimentRecord,
    NT_CSS2,
    NT_ESM,
    N_OK,
    STATUS_LABELS: dict[str, str],
    bench_summary,
    claude_candidates,
    compare,
    evidence_requirements,
    find_passages,
    gas_amount_mol,
    gas_fraction,
    run_audit,
    suggest,
    to_casefile,
    to_csv,
    to_json,
    to_markdown,
    to_review_pack,
):
    # NitroTrace interface: one anywidget. The browser draws; every number comes from the engine cells above.
    import anywidget
    import traitlets


    def scaled_record(d, k):
        """Copy of a record with the observed product multiplied by k (what-if; nondetects untouched)."""
        e = copy.deepcopy(d)
        op = e.get("observed_product")
        if op and k != 1 and not op.get("nondetect"):
            for b in ("lower", "upper"):
                if op.get(b) is not None:
                    op[b] = op[b] * k
        return e


    def audit_payload(key, base, scale):
        out = {"key": key, "scale": scale, "error": None}
        d = DATA.get(key, base) if base is None else base
        out.update(case_id=d.get("case_id", "?"), title=d.get("title", ""), record_type=d.get("record_type", "user"),
                   sources=[f"[{s}] " + DATA["protocols/sources.json"].get(s, {}).get("citation", s).split(",")[0] + " et al."
                            for s in d.get("sources", [])])
        try:
            rec = ExperimentRecord(**scaled_record(d, scale))
            res = run_audit(rec)
        except Exception as e:  # malformed edits are shown, not raised
            out.update(status="input_inconsistency", headline="The edited record could not be read.",
                       error=f"{type(e).__name__}: {e}")
            return out
        vmap = []
        if d.get("observed_product"):
            for i in range(-24, 25):
                k = 10 ** (i / 8)
                try:
                    vmap.append([k, run_audit(ExperimentRecord(**scaled_record(d, k))).status])
                except Exception:
                    vmap.append([k, "input_inconsistency"])
        req = evidence_requirements(rec, res)
        out.update(
            status=res.status, status_label=res.status_label, headline=res.headline, explanation=res.explanation,
            coverage=res.coverage, budget_label=res.budget_label, observed=res.observed_nmol,
            valid=res.budget_valid_nmol, with_scen=res.budget_with_scenarios_nmol,
            contributions=[{"label": c.label, "category": c.category, "role": c.role, "included": c.included,
                            "lower": c.lower_nmol, "upper": c.upper_nmol, "reason": c.reason} for c in res.contributions],
            unquantified=[u.model_dump() for u in res.unquantified], issues=[i.model_dump() for i in res.issues],
            recs=[{"title": r.title, "text": r.text, "trigger": r.trigger,
                   "refs": [{"quote": q["quote"], "source_id": q["source_id"], "location": q["location"]} for q in r.references]}
                  for r in res.recommendations],
            trail=[t.model_dump() for t in res.observed_trail + [s for c in res.contributions for s in c.trail]],
            assumptions=res.assumptions, req=req, vmap=vmap, hypothetical=res.hypothetical,
            whatif_fields=whatif_fields(rec, res, req),
            report={"casefile": to_casefile(rec, res, req), "review": to_review_pack(rec, res, req),
                    "md": to_markdown(rec, res), "json": to_json(rec, res), "csv": to_csv(rec, res)})
        return out


    def whatif_fields(rec, res, req):
        """Form fields for hypothetical measurements, prefilled at half of each equal-split threshold."""
        sug = {k: (v, sid) for k, v, sid in suggest(rec, req)}
        gas_ok = [g for g in rec.gas_sources if gas_amount_mol(g, rec)]
        gas_val = sug.get("gas", (None, None))[0]
        if gas_val is None and req:
            for row in req["shrink"]:
                g = next((g for g in rec.gas_sources if g.id == row["source_id"]), None)
                if g is not None and row.get("feasible") and "required_value" in row:
                    gas_val = float(f"{gas_fraction(row['required_value'], g.impurity.unit) * 1e9 * 0.9:.3g}")
        vol = rec.observed_product.volume_mL if rec.observed_product else None
        blank_kind = "electrolyte" if vol else "electrolyte_nmol"
        blank_val = sug.get("electrolyte", sug.get("electrolyte_nmol", (None, None)))[0]
        open_cat = {k for k, v in res.coverage.items() if v not in N_OK}
        return [
            {"kind": "gas", "label": "Total reactive N in the gas delivered to the cell", "unit": "ppb",
             "value": gas_val, "on": gas_val is not None, "source_id": gas_ok[0].id if gas_ok else None,
             "disabled": None if gas_ok else "Gas flow or exposure not stated in the record: no ppb level can be "
                                             "turned into an amount. State the flow first."},
            {"kind": blank_kind, "label": "Matched electrolyte blank", "unit": "µg/L NH4+" if vol else "nmol N",
             "value": blank_val, "on": "electrolyte" in open_cat and blank_val is not None, "disabled": None},
            {"kind": "components", "label": "Matched component release", "unit": "nmol N",
             "value": sug.get("components", (None, None))[0], "on": "components" in open_cat and "components" in sug,
             "disabled": None},
        ]


    def catalog():
        rows = []
        for k in sorted(CASE_KEYS, key=lambda k: (not k.startswith("cases/"), DATA[k]["title"])):
            d = DATA[k]
            try:
                st = run_audit(ExperimentRecord(**d)).status
            except Exception:
                st = "input_inconsistency"
            pub = k.startswith("cases/")
            sub = (d.get("expected") or {}).get("purpose") if not pub else ", ".join(d.get("sources", []))
            rows.append({"key": k, "id": d["case_id"], "title": d["title"], "group": "published" if pub else "synthetic",
                         "status": st, "status_label": STATUS_LABELS[st], "sub": (sub or d["case_id"])[:90]})
        return rows



    class NitroTraceApp(anywidget.AnyWidget):
        _esm = NT_ESM
        _css = NT_CSS2
        catalog = traitlets.List([]).tag(sync=True)
        bench = traitlets.Dict({}).tag(sync=True)
        selected = traitlets.Unicode("").tag(sync=True)
        scale = traitlets.Float(1.0).tag(sync=True)
        tab = traitlets.Unicode("audit").tag(sync=True)
        audit = traitlets.Dict({}).tag(sync=True)
        record_json = traitlets.Unicode("").tag(sync=True)
        edit_json = traitlets.Dict({}).tag(sync=True)
        agent_text = traitlets.Unicode("").tag(sync=True)
        agent_req = traitlets.Dict({}).tag(sync=True)
        agent_res = traitlets.Dict({}).tag(sync=True)
        whatif_req = traitlets.Dict({}).tag(sync=True)
        whatif_res = traitlets.Dict({}).tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self._base = None
            b = dict(bench_summary)
            b["elapsed_ms"] = b.pop("elapsed_s") * 1000
            b["properties"] = {k: list(v) for k, v in b["properties"].items()}
            self.bench = b
            self.catalog = catalog()
            self.selected = "cases/bragulla2026_pristine_gde_ocv.json"
            self._load()

        def _load(self):
            if self.selected not in DATA:
                return
            self._base = copy.deepcopy(DATA[self.selected])
            self.record_json = json.dumps(self._base, indent=2, ensure_ascii=False)
            self._run()

        def _run(self):
            self.whatif_res = {}
            self.audit = audit_payload(self.selected, self._base, self.scale)

        @traitlets.observe("whatif_req")
        def _on_whatif(self, change):
            r = change["new"] or {}
            changes = [(c["kind"], float(c["value"]), c.get("source_id")) for c in r.get("changes", [])]
            if not changes:
                self.whatif_res = {"error": "Tick at least one measurement."}
                return
            try:
                out = compare(scaled_record(self._base, self.scale), changes)
            except (ValueError, KeyError, TypeError) as e:
                self.whatif_res = {"error": str(e)}
                return
            out.pop("record"), out.pop("after_result")
            self.whatif_res = out

        @traitlets.observe("selected")
        def _on_selected(self, change):
            self._load()

        @traitlets.observe("scale")
        def _on_scale(self, change):
            if self._base is not None:
                self._run()

        @traitlets.observe("edit_json")
        def _on_edit(self, change):
            text = (change["new"] or {}).get("text")
            if not text:
                return
            try:
                self._base = json.loads(text)
                self.record_json = text
                self._run()
            except json.JSONDecodeError as e:
                self.audit = {**self.audit, "error": f"JSON syntax error: {e}"}

        @traitlets.observe("agent_req")
        def _on_agent(self, change):
            r = change["new"] or {}
            text = (r.get("text") or "").strip()
            if not text:
                self.agent_res = {"error": "Paste some text first."}
                return
            out = {"passages": find_passages(text)}
            if r.get("llm"):
                c = claude_candidates(text, (r.get("key") or "").strip())
                out.update(c)
            self.agent_res = out


    return (NitroTraceApp,)


@app.cell(hide_code=True)
def app(NitroTraceApp):
    # The app: everything above is hidden behind this one interface.
    nt_app = NitroTraceApp()
    mo.ui.anywidget(nt_app)

    return


if __name__ == "__main__":
    app.run()
