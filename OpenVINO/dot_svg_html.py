_dot_svg_viewer_html_template='''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 2.40.1 (20161225.0304)
 -->
<html>
<head>
    <style>
        .infobar {
            position: absolute;
            z-index: 100;
            background-color: #ffffff;
            overflow: auto;
            border: 1px solid black;
        }

        pre {
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
            font-size: 12px;
            margin: 5px;
        }

        .grabbable {
            position: absolute;
            cursor: move;
            cursor: grab;
            cursor: -moz-grab;
            cursor: -webkit-grab;
        }

        .search_li {
            cursor: pointer;
        }
        .search_li:hover {
            background-color: #ffffff;
        }
    </style>
</head>
<body>
<pre id="SearchPointer" style="position:absolute;z-index: 100;font-size:32px;color: #ff0000;pointer-events: none;">&starf;</pre>
<div id="floatbar" style="position:fixed;top:20px;right: 10px;z-index: 100;background-color: #92cff3;border: 1px solid black;padding:5px;">
    &#x1F50D;&#8239;<input type="text" id="SearchKey" placeholder="search here">&#8239;&crarr;<br>
  <div style="max-height: 450px;overflow-y: auto;">
  <ui id="SearchList">
  </ui>
  </div>
</div>

<svg></svg>

<div id="infobar" class="infobar">
    <pre id="infopre">
</pre>
</div>

<script type="text/javascript">
    infobar = document.querySelector("#infobar")
    infopre = document.querySelector("#infopre")
    SearchKey = document.querySelector("#SearchKey")
    Search = document.querySelector("#Search")
    SearchPointer = document.querySelector("#SearchPointer")
    SearchList = document.querySelector("#SearchList")
    SearchKey.onkeyup = function (ev){
        keyStr = SearchKey.value
        if (keyStr == "") {
            SearchList.innerHTML = ""
            return
        }
        if (event.keyCode !== 13) return
        SearchList.innerHTML = ""
        var elList = document.querySelectorAll('g');
        for (i = 0; i < elList.length; i++) {
            g = elList[i]
            if (g.classList.contains("graph")) continue;
            if (!(g.classList.contains("node"))) continue;
            a = g.querySelector("a")
            if (!a) continue
            title = g.querySelector("title")
            if (!title) continue
            tooltip_txt = a.getAttribute("xlink:title")
            if (!tooltip_txt) continue
            indexof = tooltip_txt.toLowerCase().indexOf(keyStr.toLowerCase())
            if (indexof >= 0) {
                li = document.createElement("li")
                li.innerHTML = title.innerHTML
                let target = { element: g };
                li.setAttribute("class", "search_li");
                var rect = g.getBoundingClientRect()
                li.setAttribute("targetRect_x", rect.left +  window.scrollX)
                li.setAttribute("targetRect_y", rect.top +  window.scrollY)
                li.onclick = function() {
                    var x = this.getAttribute("targetRect_x")
                    var y = this.getAttribute("targetRect_y")
                    window.scrollTo(x - window.innerWidth/2, y - window.innerHeight/2)
                    SearchPointer.style.top = y
                    SearchPointer.style.left = x
                }
                SearchList.appendChild(li)
            }
        }
        return;
    }
    let infobar_on = null

    infobar.addEventListener("click", function (event) {
        event.stopPropagation();
    })

    // make svg dragable
    svg = document.body.querySelector("svg")

    let pos = { top: 0, left: 0, x: 0, y: 0 };

    const mouseMoveHandler = function (e) {
        const dx = e.clientX - pos.x;
        const dy = e.clientY - pos.y;
        window.scroll(pos.left - dx, pos.top - dy);
    };

    const mouseUpHandler = function (e) {
        svg.onpointermove = null;
        svg.onpointerup = null;
        svg.releasePointerCapture(e.pointerId);
        svg.style.cursor = '';
        svg.style.removeProperty('user-select');
    };

    const mouseDownHandler = function (e, ele) {
        if (e.pointerType != "mouse") return;
        pos = {
            // The current scroll
            left: window.scrollX,
            top: window.scrollY,
            // Get the current mouse position
            x: e.clientX,
            y: e.clientY,
        };
        // Change the cursor and prevent user from selecting the text
        svg.style.cursor = 'grabbing';
        svg.style.userSelect = 'none';
        svg.setPointerCapture(e.pointerId);
        svg.onpointermove = mouseMoveHandler;
        svg.onpointerup = mouseUpHandler;
    };

    svg.classList.add("grabbable");
    svg.onpointerdown = mouseDownHandler;

    var elList = document.querySelectorAll('g');
    elList.forEach(
        function (g) {
            if (g.classList.contains("graph")) return;
            a = g.querySelector("a")
            if (!a) return
            tooltip_txt = a.getAttribute("xlink:title")
            if (!tooltip_txt) return
            g.style.cursor = "pointer"

            // prevent `pointerdown` being processed by `mouseDownHandler`
            // which in turn capture the pointer and fails to trigger g.onclick()
            g.onpointerdown = function (event) { event.stopPropagation(); }

            // pop up info bar
            g.onclick = function (event) {
                // toggle
                if (infobar_on === this) {
                    infobar_on = null
                    infobar.style.display = "none";
                } else {
                    infobar_on = this;
                    a = this.querySelector("a")
                    tooltip_txt = a.getAttribute("xlink:title")
                    var rect = this.getBoundingClientRect()
                    infopre.innerHTML = tooltip_txt
                    infobar.style.top = rect.top + window.scrollY
                    infobar.style.left = rect.right + window.scrollX
                    // remove custom setting, recover display's orginal setting in CSS
                    infobar.style.display = '';
                }
                
                if (event)
                    event.stopPropagation();
            }
        }
    )
</script>

</body>
</html>
'''

import re

def dot_to_html(dot_source, htmlfilepath = None):
    m = re.search('(<svg(.|\n)*<\/svg>)', dot_source)
    output_src = _dot_svg_viewer_html_template.replace("<svg></svg>", m.group(1))
    if htmlfilepath:
        with open(htmlfilepath,'w') as f:
            f.write(output_src)
            f.close()
    return output_src