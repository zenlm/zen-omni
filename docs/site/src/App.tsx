import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import DocumentationPage from './pages/DocumentationPage'
import LanguagePage from './pages/LanguagePage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/docs" element={<DocumentationPage />} />
        <Route path="/docs/:language" element={<LanguagePage />} />
      </Routes>
    </Layout>
  )
}

export default App