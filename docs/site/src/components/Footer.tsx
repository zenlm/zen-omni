import { LuxIcon } from './LuxLogo'

export default function Footer() {
  return (
    <footer className="border-t-4 border-black py-12 mt-24 bg-black text-white">
      <div className="container">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <div className="flex items-center space-x-3">
            <LuxIcon className="w-6 h-6 text-white" />
            <span className="font-mono text-sm text-gray-400 uppercase tracking-wide">
              © 2025 Lux Network
            </span>
          </div>
          
          <div className="flex items-center space-x-8">
            <a 
              href="https://lux.network" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sm text-gray-400 hover:text-white transition-colors font-medium uppercase tracking-wide"
            >
              Lux Network
            </a>
            <a 
              href="https://hanzo.ai" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sm text-gray-400 hover:text-white transition-colors font-medium uppercase tracking-wide"
            >
              Hanzo AI
            </a>
            <a 
              href="https://github.com/luxfi" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sm text-gray-400 hover:text-white transition-colors font-medium uppercase tracking-wide"
            >
              GitHub
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}