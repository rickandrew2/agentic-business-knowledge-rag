# Test file upload to RAG system

$file = "C:\Users\Acer\Desktop\A4Coding\AgenticAI-RAG\rag-assistant\data\sample-sales.csv"
$uri = "http://localhost:8000/api/data/upload"

# Read file content
$fileContent = [System.IO.File]::ReadAllBytes($file)
$fileName = [System.IO.Path]::GetFileName($file)

# Create multipart form data
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

$bodyLines = @(
    "--$boundary",
    "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"",
    "Content-Type: text/csv",
    "",
    [System.Text.Encoding]::UTF8.GetString($fileContent),
    "--$boundary--"
) -join $LF

try {
    $response = Invoke-WebRequest -Uri $uri -Method POST -ContentType "multipart/form-data; boundary=$boundary" -Body $bodyLines -UseBasicParsing
    Write-Host "Success: $($response.StatusCode)"
    Write-Host $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 3
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    Write-Host $_.Exception.Response
}
