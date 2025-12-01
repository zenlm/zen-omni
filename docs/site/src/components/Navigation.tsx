import { Link } from 'react-router-dom'
import { Menu, X, Github, Book, Code2 } from 'lucide-react'
import { useState } from 'react'
import { Button } from './ui/button'
import LuxLogo from './LuxLogo'

export default function Navigation() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="border-b-4 border-black sticky top-0 z-50 bg-white">
      <div className="container">
        <div className="flex h-20 items-center justify-between">
          {/* Logo */}
          <LuxLogo 
            href="/" 
            size="md"
            variant="full"
            outerClx="no-underline"
            textClx="text-black font-heading uppercase"
          />

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center space-x-8">
            <Link to="/docs" className="text-sm font-medium text-black hover:text-gray-600 transition-colors uppercase tracking-wide">
              Documentation
            </Link>
            <Link to="/docs#languages" className="text-sm font-medium text-black hover:text-gray-600 transition-colors uppercase tracking-wide">
              Implementations
            </Link>
            <Button variant="outline" size="sm" asChild>
              <a 
                href="https://github.com/luxfi/consensus" 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center"
              >
                <Github className="w-4 h-4 mr-2" />
                GitHub
              </a>
            </Button>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden p-2"
          >
            {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Nav */}
        {isOpen && (
          <div className="md:hidden py-4 border-t border-gray-200">
            <div className="flex flex-col space-y-4">
              <Link 
                to="/docs" 
                className="flex items-center space-x-2 py-2 text-gray-600 hover:text-black transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <Book className="w-4 h-4" />
                <span className="font-medium">Documentation</span>
              </Link>
              <Link 
                to="/docs#languages" 
                className="flex items-center space-x-2 py-2 text-gray-600 hover:text-black transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <Code2 className="w-4 h-4" />
                <span className="font-medium">Languages</span>
              </Link>
              <a 
                href="https://github.com/luxfi/consensus" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center space-x-2 py-2 text-gray-600 hover:text-black transition-colors"
              >
                <Github className="w-4 h-4" />
                <span className="font-medium">GitHub</span>
              </a>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}