package templates

var BaseTpl = `
{{- define "base" }}
<div class="container">
    <div class="item" id="{{ .ChartID }}"
         style="width:{{ .InitOpts.Width }};height:{{ .InitOpts.Height }};"></div>
</div>
<script type="text/javascript">
    "use strict";
    var myChart___x__{{ .ChartID }}__x__ = echarts.init(document.getElementById('{{ .ChartID }}'), "{{ .Theme }}");
    var option___x__{{ .ChartID }}__x__ = {
        title: {{ .TitleOpts  }},
        tooltip: {{ .TooltipOpts }},
        legend: {{ .LegendOpts }},
    {{- if .HasGeo }}
        geo: {{ .GeoComponentOpts }},
    {{- end }}
    {{- if .HasRadar }}
        radar: {{ .RadarComponentOpts }},
    {{- end }}
    {{- if .HasParallel }}
        parallel: {{ .ParallelComponentOpts }},
        parallelAxis: {{ .ParallelAxisOpts }},
    {{- end }}
    {{- if .HasSingleAxis }}
        singleAxis: {{ .SingleAxisOpts }},
    {{- end }}
    {{- if .ToolboxOpts.Show }}
        toolbox: {{ .ToolboxOpts }},
    {{- end }}
    {{- if gt .DataZoomOptsList.Len 0 }}
        dataZoom:{{ .DataZoomOptsList }},
    {{- end }}
    {{- if gt .VisualMapOptsList.Len 0 }}
        visualMap:{{ .VisualMapOptsList }},
    {{- end }}
    {{- if .HasXYAxis }}
        xAxis: {{ .XAxisOptsList }},
        yAxis: {{ .YAxisOptsList }},
    {{- end }}
    {{- if .Has3DAxis }}
        xAxis3D: {{ .XAxis3D }},
        yAxis3D: {{ .YAxis3D }},
        zAxis3D: {{ .ZAxis3D }},
        grid3D: {{ .Grid3D }},
    {{- end }}
        series: [
        {{ range .Series }}
        {{- . }},
        {{ end -}}
        ],
    {{- if eq .Theme "white" }}
        color: {{ .Colors }},
    {{- end }}
    {{- if ne .BackgroundColor "" }}
        backgroundColor: {{ .BackgroundColor }}
    {{- end }}
    };
    myChart___x__{{ .ChartID }}__x__.setOption(option___x__{{ .ChartID }}__x__);

    {{- range .JSFunctions.Fns }}
		{{ . }}
	{{- end }}
</script>
{{ end }}
`
