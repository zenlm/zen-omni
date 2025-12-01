import * as React from "react"
import { cn } from "../../lib/utils"

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "link"
  size?: "default" | "sm" | "lg" | "icon"
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
    const Comp = asChild ? "span" : "button"
    
    return (
      <Comp
        className={cn(
          "inline-flex items-center justify-center whitespace-nowrap text-sm font-black transition-all focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-black focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 uppercase tracking-wider",
          {
            "bg-black text-white hover:bg-white hover:text-black border-2 border-black": variant === "default",
            "border-2 border-black bg-white hover:bg-black hover:text-white text-black": variant === "outline",
            "hover:bg-black hover:text-white": variant === "ghost",
            "text-black underline-offset-4 hover:underline": variant === "link",
          },
          {
            "h-12 px-6 py-2": size === "default",
            "h-10 px-4": size === "sm",
            "h-14 px-10": size === "lg",
            "h-12 w-12": size === "icon",
          },
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button }