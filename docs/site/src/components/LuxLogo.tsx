import React from 'react'
import { Link } from 'react-router-dom'

type TShirtSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl'
type LogoVariant = 'text-only' | 'logo-only' | 'full'

export const LuxIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50" {...props}>
    <polygon points="25,46.65 50,3.35 0,3.35" fill="currentColor" />
  </svg>
)

const LuxLogo: React.FC<{
  size?: TShirtSize
  variant?: LogoVariant
  onClick?: () => void
  href?: string
  outerClx?: string
  textClx?: string
}> = ({
  size = 'md',
  href,
  outerClx = '',
  textClx = '',
  variant = 'full',
  onClick,
}) => {
  let classes: any = {}
  const toAdd = (variant === 'logo-only') ? 
    {
      span: 'hidden',
      icon: ''
    }  
    :
    (variant === 'text-only') ? 
      {
        span: '',
        icon: 'hidden'
      } 
      : 
      {
        span: '',
        icon: ''
      }

  if (size === 'lg' || size === 'xl' ) {
    classes.icon = 'h-10 w-10 mr-4' 
    classes.span = 'text-3xl' 
  }
  else if (size === 'md') {
    classes.icon = 'h-8 w-8 mr-3'
    classes.span = 'text-2xl tracking-tighter'
  }
  else if (size === 'sm' ) {
    classes.icon = 'h-6 w-6 mr-2' 
    classes.span = 'text-lg'
  }
  else { // xs
    classes.icon = 'h-4 w-4 mr-1'
    classes.span = 'text-base'
  }

  classes.icon += ' ' + toAdd.icon
  classes.span += ' ' + toAdd.span

  const outerClasses = 'flex flex-row items-center ' + outerClx
  const spanClasses = 'inline-block font-bold ' 
    + textClx
    + (href ? ' hover:text-gray-600 cursor-pointer ' : ' cursor-default ') 
    + classes.span 

  return (
    href ? (
      <Link to={href} className={outerClasses} onClick={onClick} >
        <LuxIcon className={classes.icon} />
        <span className={spanClasses}>LUX</span>
      </Link>
    ) : (
      <span className={outerClasses} onClick={onClick}>
        <LuxIcon className={classes.icon} />
        <span className={spanClasses}>LUX</span>
      </span>
    )
  )
}

export default LuxLogo