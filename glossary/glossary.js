// ========================================
// GLOSSARY ENGINE - Dynamic Markdown Parser
// ========================================

const GlossaryApp = {
    entries: [],
    allTags: new Set(),
    currentLang: 'de',
    activeFilters: {
        search: '',
        category: 'all',
        levels: new Set(['basic', 'intermediate', 'advanced'])
    },

    // Initialize the application
    async init() {
        await this.loadGlossary();
        this.setupEventListeners();
        this.handleUrlHash();
    },

    // Load and parse glossary markdown
    async loadGlossary() {
        const filename = this.currentLang === 'de' ? 'glossar_de.md' : 'glossary_en.md';

        try {
            const response = await fetch(filename);
            if (!response.ok) throw new Error(`Failed to load ${filename}`);

            const markdown = await response.text();
            this.entries = this.parseMarkdown(markdown);
            this.updateUI();
        } catch (error) {
            console.error('Error loading glossary:', error);
            document.getElementById('glossary-entries').innerHTML =
                '<div class="loading-state"><p>Fehler beim Laden des Glossars.</p></div>';
        }
    },

    // Parse markdown into structured entries
    parseMarkdown(markdown) {
        const entries = [];
        const blocks = markdown.split(/\n(?=## )/);

        blocks.forEach(block => {
            if (!block.trim() || !block.startsWith('## ')) return;
            if (block.startsWith('## Ressourcen') || block.startsWith('# ')) return;

            const lines = block.split('\n');
            const entry = {
                title: lines[0].replace(/^## /, '').trim(),
                id: '',
                en: '',
                tags: [],
                level: 'basic',
                body: '',
                sources: []
            };

            let bodyStart = 1;

            // Parse metadata
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();

                if (line.startsWith('id:')) {
                    entry.id = line.replace('id:', '').trim();
                } else if (line.startsWith('en:')) {
                    entry.en = line.replace('en:', '').trim();
                } else if (line.startsWith('tags:')) {
                    entry.tags = line.replace('tags:', '').split(',').map(t => t.trim());
                    // Add to allTags but exclude 'wip' (it's a status marker, not a category)
                    entry.tags.forEach(tag => {
                        if (tag !== 'wip') this.allTags.add(tag);
                    });
                } else if (line.startsWith('level:')) {
                    entry.level = line.replace('level:', '').trim();
                } else if (line === '') {
                    bodyStart = i + 1;
                    break;
                }
            }

            // Parse body and sources
            const remaining = lines.slice(bodyStart);
            const bodyLines = [];
            const sourceLines = [];

            remaining.forEach(line => {
                if (line.trim().startsWith('* ')) {
                    sourceLines.push(line.trim().substring(2));
                } else if (line.trim() && !line.startsWith('#')) {
                    bodyLines.push(line.trim());
                }
            });

            entry.body = bodyLines.join('\n\n');
            entry.sources = sourceLines;

            if (entry.id && entry.title) {
                entries.push(entry);
            }
        });

        // Sort alphabetically
        return entries.sort((a, b) => a.title.localeCompare(b.title, 'de'));
    },

    // Update the entire UI
    updateUI() {
        this.renderEntries();
        this.updateStats();
        this.updateAlphabet();
        this.updateCategories();
        this.updateLevels();
    },

    // Render filtered entries grouped by letter
    renderEntries() {
        const container = document.getElementById('glossary-entries');
        const filtered = this.getFilteredEntries();

        if (filtered.length === 0) {
            container.innerHTML = '<div class="loading-state"><p>Keine Einträge gefunden.</p></div>';
            return;
        }

        // Group by first letter
        const grouped = {};
        filtered.forEach(entry => {
            const letter = entry.title.charAt(0).toUpperCase();
            if (!grouped[letter]) grouped[letter] = [];
            grouped[letter].push(entry);
        });

        let html = '';
        Object.keys(grouped).sort().forEach(letter => {
            html += this.renderLetterSection(letter, grouped[letter]);
        });

        container.innerHTML = html;

        // Add click handlers for expand/collapse
        container.querySelectorAll('.term-header').forEach(header => {
            header.addEventListener('click', () => {
                header.closest('.term-card').classList.toggle('expanded');
            });
        });
    },

    // Render a letter section with entries
    renderLetterSection(letter, entries) {
        return `
            <section class="letter-section" id="letter-${letter}">
                <div class="letter-heading">
                    <div class="letter-char">${letter}</div>
                    <div class="letter-meta">${entries.length} ${entries.length === 1 ? 'Begriff' : 'Begriffe'}</div>
                </div>
                ${entries.map(e => this.renderEntryCard(e)).join('')}
            </section>
        `;
    },

    // Render a single entry card
    renderEntryCard(entry) {
        const preview = this.getPreview(entry.body);
        const bodyHtml = this.formatBody(entry.body, entry.id);
        const sourcesHtml = entry.sources.length > 0 ? `
            <div class="term-sources">
                <h4>Quellen</h4>
                <ul>
                    ${entry.sources.map(s => `<li>${this.formatSource(s)}</li>`).join('')}
                </ul>
            </div>
        ` : '';


        return `
            <article class="term-card" id="${entry.id}">
                <div class="term-header">
                    <div class="term-title-group">
                        <h3 class="term-title">${entry.title}</h3>
                        ${entry.tags.length > 0 ? `<div class="term-categories">${entry.tags.map(t => `<span class="term-category" data-tag="${t}">${this.formatTagDisplay(t)}</span>`).join('')}</div>` : ''}
                    </div>
                    <span class="term-level ${entry.level}">${this.capitalizeFirst(entry.level)}</span>
                    <div class="expand-toggle">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </div>
                </div>
                <p class="term-preview">${preview}</p>
                <div class="term-body">
                    <div class="term-definition">${bodyHtml}</div>
                    ${sourcesHtml}
                </div>
            </article>
        `;
    },

    // Get preview text (first 2 sentences or ~250 chars)
    getPreview(body) {
        const text = body.replace(/\[\[#[^\]]+\|([^\]]+)\]\]/g, '$1')
                         .replace(/\[\[#([^\]]+)\]\]/g, '$1')
                         .replace(/\*\*([^*]+)\*\*/g, '$1')
                         .replace(/_([^_]+)_/g, '$1');

        // Split into sentences and take first 2
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        let preview = sentences.slice(0, 2).join(' ').trim();

        // If still too short, add more; if too long, truncate
        if (preview.length > 300) {
            preview = preview.substring(0, 280).trim() + '...';
        } else if (preview.length < 80 && sentences.length > 2) {
            preview = sentences.slice(0, 3).join(' ').trim();
        }

        return preview || text.substring(0, 200) + '...';
    },

    // Format body text with links and styling
    formatBody(body, currentEntryId = '') {
        let html = body
            // Convert [[#id|text]] to links (skip self-references)
            .replace(/\[\[#([^\]|]+)\|([^\]]+)\]\]/g, (match, id, text) => {
                // Find entry by id or title to get the correct id
                const entry = this.entries.find(e => e.id === id || e.title === id || e.title.toLowerCase() === id.toLowerCase());
                const targetId = entry ? entry.id : id;
                if (targetId === currentEntryId) return text; // No link for self-reference
                return `<a href="#${targetId}" class="term-link">${text}</a>`;
            })
            // Convert [[#id]] to links (skip self-references)
            .replace(/\[\[#([^\]]+)\]\]/g, (match, id) => {
                const entry = this.entries.find(e => e.id === id || e.title === id || e.title.toLowerCase() === id.toLowerCase());
                const text = entry ? entry.title : id;
                const targetId = entry ? entry.id : id;
                // Skip self-references
                if (targetId === currentEntryId) return text;
                return `<a href="#${targetId}" class="term-link">${text}</a>`;
            })
            // Bold text
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            // Italic text
            .replace(/_([^_]+)_/g, '<em>$1</em>')
            // Paragraphs
            .split('\n\n')
            .map(p => `<p>${p}</p>`)
            .join('');

        return html;
    },

    // Format source citations with icons
    formatSource(source) {
        // Determine icon based on URL or content
        let icon = '';
        const lowerSource = source.toLowerCase();

        if (lowerSource.includes('youtube.com') || lowerSource.includes('youtu.be')) {
            icon = '<i class="fab fa-youtube source-icon youtube"></i>';
        } else if (lowerSource.includes('arxiv.org') || lowerSource.includes('arxiv.')) {
            icon = '<i class="fas fa-file-alt source-icon arxiv"></i>';
        } else if (lowerSource.includes('github.com') || lowerSource.includes('github.io')) {
            icon = '<i class="fab fa-github source-icon github"></i>';
        } else if (lowerSource.includes('deeplearningbook') || lowerSource.includes('mit press') || lowerSource.includes('_book_') || source.includes('MIT Press') || source.includes('Springer') || source.includes('O\'Reilly')) {
            icon = '<i class="fas fa-book source-icon book"></i>';
        } else if (lowerSource.includes('doi.org') || lowerSource.includes('acm.org') || lowerSource.includes('ieee.org') || lowerSource.includes('aclanthology') || lowerSource.includes('proceedings') || lowerSource.includes('journal') || lowerSource.includes('conference')) {
            icon = '<i class="fas fa-scroll source-icon paper"></i>';
        } else {
            icon = '<i class="fas fa-link source-icon web"></i>';
        }

        // Convert markdown to HTML
        let formatted = source;
        // Convert markdown links [text](url) to HTML
        formatted = formatted.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
        // Convert markdown italic _text_ to HTML <em>
        formatted = formatted.replace(/_([^_]+)_/g, '<em>$1</em>');
        return icon + formatted;
    },

    // Get filtered entries based on current filters
    getFilteredEntries() {
        return this.entries.filter(entry => {
            // Search filter
            if (this.activeFilters.search) {
                const search = this.activeFilters.search.toLowerCase();
                const match = entry.title.toLowerCase().includes(search) ||
                             entry.en.toLowerCase().includes(search) ||
                             entry.body.toLowerCase().includes(search) ||
                             entry.tags.some(t => t.toLowerCase().includes(search));
                if (!match) return false;
            }

            // Category filter
            if (this.activeFilters.category !== 'all') {
                if (!entry.tags.includes(this.activeFilters.category)) return false;
            }

            // Level filter
            if (!this.activeFilters.levels.has(entry.level)) return false;

            return true;
        });
    },

    // Update statistics
    updateStats() {
        const filtered = this.getFilteredEntries();
        document.getElementById('stat-terms').textContent = filtered.length;
        document.getElementById('stat-categories').textContent = this.allTags.size;
    },

    // Update alphabet navigation
    updateAlphabet() {
        const usedLetters = new Set(this.entries.map(e => e.title.charAt(0).toUpperCase()));

        document.querySelectorAll('.alpha-btn').forEach(btn => {
            const letter = btn.textContent.trim();
            if (usedLetters.has(letter)) {
                btn.classList.remove('disabled');
                btn.href = `#letter-${letter}`;
            } else {
                btn.classList.add('disabled');
                btn.removeAttribute('href');
            }
        });
    },

    // Update category list
    updateCategories() {
        const container = document.querySelector('.category-list');
        const tagCounts = {};

        this.entries.forEach(entry => {
            entry.tags.forEach(tag => {
                // Exclude 'wip' from categories - it's a status marker, not a category
                if (tag !== 'wip') {
                    tagCounts[tag] = (tagCounts[tag] || 0) + 1;
                }
            });
        });

        const sortedTags = Object.entries(tagCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        let html = `
            <div class="category-item active" data-category="all">
                <span>Alle</span>
                <span class="count">${this.entries.length}</span>
            </div>
        `;

        sortedTags.forEach(([tag, count]) => {
            html += `
                <div class="category-item" data-category="${tag}">
                    <span>${this.formatTagDisplay(tag)}</span>
                    <span class="count">${count}</span>
                </div>
            `;
        });

        container.innerHTML = html;

        // Re-attach event listeners
        container.querySelectorAll('.category-item').forEach(item => {
            item.addEventListener('click', () => {
                container.querySelectorAll('.category-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                this.activeFilters.category = item.dataset.category;
                this.renderEntries();
                this.updateStats();
            });
        });
    },

    // Update level filters
    updateLevels() {
        const levelCounts = { basic: 0, intermediate: 0, advanced: 0 };
        this.entries.forEach(e => levelCounts[e.level]++);

        document.querySelectorAll('.level-item').forEach(item => {
            const level = item.dataset.level;
            const countEl = item.querySelector('span:last-child');
            if (countEl && levelCounts[level] !== undefined) {
                countEl.textContent = levelCounts[level];
            }
        });
    },

    // Setup event listeners
    setupEventListeners() {
        // Search input
        const searchInput = document.getElementById('search');
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.activeFilters.search = e.target.value;
                this.renderEntries();
                this.updateStats();
            }, 200);
        });

        // Keyboard shortcut for search
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                searchInput.focus();
            }
        });

        // Level filters
        document.querySelectorAll('.level-item').forEach(item => {
            item.addEventListener('click', () => {
                item.classList.toggle('active');
                const level = item.dataset.level;

                if (item.classList.contains('active')) {
                    this.activeFilters.levels.add(level);
                } else {
                    this.activeFilters.levels.delete(level);
                }

                this.renderEntries();
                this.updateStats();
            });
        });

        // Language toggle
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const newLang = btn.textContent.trim().toLowerCase();
                if (newLang !== this.currentLang) {
                    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.currentLang = newLang;
                    this.allTags.clear();
                    await this.loadGlossary();
                }
            });
        });

        // Smooth scroll for alphabet
        document.querySelectorAll('.alpha-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                if (!btn.classList.contains('disabled')) {
                    e.preventDefault();
                    const target = document.querySelector(btn.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }
            });
        });

        // Handle clicks on internal links
        document.addEventListener('click', (e) => {
            // Use closest() to handle clicks on child elements within links
            const termLink = e.target.closest('.term-link, .related-link');
            if (termLink) {
                e.preventDefault();
                const id = termLink.getAttribute('href').substring(1);
                this.navigateToEntry(id);
            }
            // Handle clicks on category tags - filter by that tag
            if (e.target.classList.contains('term-category')) {
                e.stopPropagation(); // Don't toggle card expansion
                const tag = e.target.dataset.tag;
                this.filterByTag(tag);
            }
        });

        // Handle hashchange for back/forward navigation
        window.addEventListener('hashchange', () => {
            if (window.location.hash) {
                const id = window.location.hash.substring(1);
                this.scrollToEntry(id);
            }
        });
    },

    // Navigate to an entry - resets filters if needed and scrolls to it
    navigateToEntry(id) {
        // First check if the entry exists in our data
        const entry = this.entries.find(e => e.id === id);
        if (!entry) {
            console.warn(`Entry not found: ${id}`);
            return;
        }

        // Check if element is currently in the DOM
        let element = document.getElementById(id);

        if (!element) {
            // Entry exists but is filtered out - reset filters to show it
            this.activeFilters.search = '';
            this.activeFilters.category = 'all';
            this.activeFilters.levels = new Set(['basic', 'intermediate', 'advanced']);

            // Update UI elements
            document.getElementById('search').value = '';
            document.querySelectorAll('.category-item').forEach(item => {
                item.classList.toggle('active', item.dataset.category === 'all');
            });
            document.querySelectorAll('.level-item').forEach(item => {
                item.classList.add('active');
            });

            // Re-render entries
            this.renderEntries();
            this.updateStats();

            // Now get the element
            element = document.getElementById(id);
        }

        if (element) {
            this.scrollToEntry(id);
        }
    },

    // Scroll to and highlight an entry
    scrollToEntry(id) {
        const element = document.getElementById(id);
        if (element) {
            // Only apply effects for actual entries, not letter sections
            const isEntry = !id.startsWith('letter-');

            if (isEntry) {
                // Update URL
                history.pushState(null, null, `#${id}`);
                // Expand the card
                element.classList.add('expanded');
            }

            // Scroll with offset for fixed header
            const headerOffset = 90;
            const elementPosition = element.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });

            // Highlight effect only for entries
            if (isEntry) {
                element.style.boxShadow = '0 0 0 3px var(--primary-light)';
                setTimeout(() => {
                    element.style.boxShadow = '';
                }, 2000);
            }
        }
    },

    // Handle URL hash on page load
    handleUrlHash() {
        if (window.location.hash) {
            const id = window.location.hash.substring(1);
            setTimeout(() => this.scrollToEntry(id), 300);
        }
    },

    // Filter by specific tag (called when clicking a tag)
    filterByTag(tag) {
        // Update filter
        this.activeFilters.category = tag;

        // Update category list UI
        document.querySelectorAll('.category-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.category === tag) {
                item.classList.add('active');
            }
        });

        // Scroll to top of entries
        document.querySelector('.glossary-content').scrollIntoView({ behavior: 'smooth' });

        // Re-render
        this.renderEntries();
        this.updateStats();
    },

    // Format tag display name (ai-engineering -> AI-Engineering, etc.)
    formatTagDisplay(tag) {
        const displayNames = {
            'ai-engineering': 'AI-Engineering',
            'fundamentals': 'Fundamentals',
            'architecture': 'Architecture',
            'training': 'Training',
            'prompting': 'Prompting',
            'agents': 'Agents',
            'safety': 'Safety',
            'evaluation': 'Evaluation',
            'governance': 'Governance',
            'benchmarks': 'Benchmarks'
        };
        return displayNames[tag] || this.capitalizeFirst(tag);
    },

    // Utility: capitalize first letter
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    GlossaryApp.init();
});
