import { useState, useRef } from 'react'
import { Upload, Activity, AlertCircle, ShieldCheck, CheckCircle2, X, Download } from 'lucide-react'
import './index.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResult(null)
      setError(null)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResult(null)
      setError(null)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const handleSubmit = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to process image')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (level) => {
    switch (level) {
      case 'Low': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50'
      case 'Medium': return 'bg-amber-500/20 text-amber-400 border-amber-500/50'
      case 'High': return 'bg-rose-500/20 text-rose-400 border-rose-500/50'
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/50'
    }
  }

  const getRiskIcon = (level) => {
    switch (level) {
      case 'Low': return <ShieldCheck className="w-5 h-5 text-emerald-400" />
      case 'Medium': return <AlertCircle className="w-5 h-5 text-amber-400" />
      case 'High': return <Activity className="w-5 h-5 text-rose-400" />
      default: return <CheckCircle2 className="w-5 h-5 text-slate-400" />
    }
  }

  const handleDownloadReport = () => {
    if (!result) return;
    
    const reportContent = `
=========================================
      VISION CARE - DIAGNOSTIC REPORT
=========================================
Condition: ${result.prediction}
Confidence: ${result.confidence}
Risk Level: ${result.risk_level}

RECOMMENDATION:
${result.recommendation}
=========================================
Report generated automatically by Vision Care.
Disclaimer: This is an AI analysis and should be verified by a medical professional.
    `.trim();

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `VisionCare_Report_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-container">
      {/* Background elements */}
      <div className="bg-blob blob-1"></div>
      <div className="bg-blob blob-2"></div>
      
      <main className="main-content">
        <header className="header">
          <h1>Vision Care</h1>
          <p>Advanced Retinal Image Analysis using PyTorch & Grad-CAM</p>
        </header>

        <div className="content-grid">
          {/* Upload Section */}
          <div className="glass-panel upload-section">
            <h2 className="panel-title">Upload Image</h2>
            
            {!previewUrl ? (
              <div 
                className="drop-zone"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current.click()}
              >
                <div className="drop-zone-content">
                  <Upload className="upload-icon" />
                  <p className="primary-text">Drag & drop your retinal scan here</p>
                  <p className="secondary-text">or click to browse from your computer</p>
                </div>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileSelect} 
                  accept="image/*" 
                  className="hidden-input"
                />
              </div>
            ) : (
              <div className="preview-container">
                <button 
                  className="clear-btn"
                  onClick={() => {
                    setSelectedFile(null)
                    setPreviewUrl(null)
                    setResult(null)
                  }}
                >
                  <X className="w-5 h-5" />
                </button>
                <img src={previewUrl} alt="Preview" className="preview-image" />
              </div>
            )}

            <button 
              className={`analyze-btn ${!selectedFile || loading ? 'disabled' : ''}`}
              onClick={handleSubmit}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  Analyzing Scan...
                </>
              ) : (
                'Run AI Diagnostics'
              )}
            </button>

            {error && (
              <div className="error-message">
                <AlertCircle className="w-5 h-5" />
                <p>{error}</p>
              </div>
            )}
          </div>

          {/* Results Section */}
          {result && (
            <div className="glass-panel results-section slide-in">
              <h2 className="panel-title">Analysis Results</h2>
              
              <div className="results-grid">
                <div className="result-metric">
                  <span className="metric-label">Detected Condition</span>
                  <span className="metric-value primary">{result.prediction}</span>
                </div>
                
                <div className="result-metric">
                  <span className="metric-label">AI Confidence</span>
                  <span className="metric-value">{result.confidence}</span>
                </div>
              </div>

              <div className={`risk-badge ${getRiskColor(result.risk_level)}`}>
                {getRiskIcon(result.risk_level)}
                <div className="risk-text">
                  <span className="risk-label">Risk Level: {result.risk_level}</span>
                  <span className="risk-recommendation">{result.recommendation}</span>
                </div>
              </div>

              <div className="heatmap-container">
                <h3 className="heatmap-title">Grad-CAM Heatmap Analysis</h3>
                <p className="heatmap-desc">Areas highlighted in red/yellow indicate the precise regions the AI focused on to make its diagnostic decision.</p>
                <div className="heatmap-image-wrapper">
                  <img 
                    src={`http://localhost:8000${result.heatmap_url}`} 
                    alt="Grad-CAM Heatmap" 
                    className="heatmap-image"
                  />
                </div>
              </div>
              
              <button 
                className="analyze-btn" 
                style={{ marginTop: '20px', background: 'rgba(14, 165, 233, 0.2)', borderColor: 'rgba(14, 165, 233, 0.4)' }}
                onClick={handleDownloadReport}
              >
                <Download className="w-5 h-5 mr-2 inline" />
                Download Diagnostic Report
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
