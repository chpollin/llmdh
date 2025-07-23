# DESIGN.md

## Design Principles
- **Scholarly Minimalism**: Function over form
- **Information Density**: Maximum content, minimum decoration
- **Print-First**: Optimized for both screen and paper
- **Accessibility**: WCAG AA compliant, no unnecessary animations

## Layout Structure
```
HEADER (fixed)
├── DHCraft Logo (left)
├── Page Title (center)
└── Navigation Menu (right)

MAIN CONTENT
├── Hero Section (brief intro + overview table)
├── Table of Contents (sticky sidebar on desktop)
├── Section 1: LLM Fundamentals
├── Section 2: Prompt Engineering
└── Section 3: LLM-Supported Modeling

FOOTER
├── Contact Information
├── DHCraft Logo
└── Copyright
```

## Typography
- **Headings**: Georgia or Crimson Text (serif)
- **Body**: Arial or Helvetica (system fonts)
- **Size**: 16px base, 1.6 line-height
- **Hierarchy**: Clear size distinctions (h1: 2.5em, h2: 1.8em, h3: 1.3em)

## Colors
- **Background**: #FFFFFF
- **Text**: #1a1a1a
- **Accent**: #2c5aa0 (links, section numbers)
- **Borders**: #e0e0e0
- **Code/Pre**: #f5f5f5 background

## Components

### Section Layout
- Number badge (left margin)
- Title + description
- Schedule table
- Resources list (icon + title + link)
- Collapsible subsections (simple +/- toggle)

### Resource Items
```
[Icon] Resource Title
       Type: Slides/Doc/Reference
       → Link
```

### Schedule Tables
- Simple HTML tables
- Alternating row colors (#f9f9f9)
- Time | Topic | Resources columns

## Responsive Behavior
- **Desktop** (>1024px): Sidebar TOC + main content
- **Tablet** (768-1024px): Full width, no sidebar
- **Mobile** (<768px): Stacked layout, hamburger menu

## Interactive Elements
- Links: underline on hover
- Sections: expand/collapse for subsections
- No animations except essential transitions
- Focus states for keyboard navigation

## Special Considerations
- Print stylesheet removes navigation, adds page breaks
- High contrast mode support
- Sans-serif fallbacks for all fonts
- Maximum width: 1200px centered