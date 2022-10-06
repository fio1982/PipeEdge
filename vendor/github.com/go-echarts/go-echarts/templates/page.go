package templates

var PageTpl = `
{{- define "page" }}
<!DOCTYPE html>
<html>
{{- template "header" . }}
<body>
{{- template "routers" . }}
{{- range .Charts }}
    {{ template "base" . }}
    <br/>
{{- end }}
<style>
    .container {display: flex;justify-content: center;align-items: center;}
    .item {margin: auto;}
</style>
</body>
</html>
{{ end }}
`
