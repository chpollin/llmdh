// ========================================
// GLOSSARY ENGINE - Dynamic Markdown Parser
// ========================================

const GlossaryApp = {
    // Configuration
    config: {
        debounceMs: 200,
        headerOffset: 90,
        highlightDurationMs: 2000,
        hashScrollDelayMs: 300,
        maxCategories: 10
    },

    // Localization strings
    i18n: {
        de: {
            term: 'Begriff',
            terms: 'Begriffe',
            sources: 'Quellen',
            noResults: 'Keine Einträge gefunden.',
            loadError: 'Fehler beim Laden des Glossars.',
            loading: 'Glossar wird geladen...',
            contested: 'Umstrittener Begriff'
        },
        en: {
            term: 'Term',
            terms: 'Terms',
            sources: 'Sources',
            noResults: 'No entries found.',
            loadError: 'Error loading glossary.',
            loading: 'Loading glossary...',
            contested: 'Contested concept'
        }
    },

    // Tag display names
    tagDisplayNames: {
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
    },

    // DOM element cache
    dom: {},

    // State
    entries: [],
    allTags: new Set(),
    currentLang: 'de',
    activeFilters: {
        search: '',
        category: 'all',
        levels: new Set(['basic', 'intermediate', 'advanced'])
    },

    // Get localized string
    t(key) {
        return this.i18n[this.currentLang]?.[key] || this.i18n.de[key] || key;
    },

    // Cache DOM elements
    cacheDOM() {
        this.dom = {
            entriesContainer: document.getElementById('glossary-entries'),
            searchInput: document.getElementById('search'),
            statTerms: document.getElementById('stat-terms'),
            statCategories: document.getElementById('stat-categories'),
            categoryList: document.querySelector('.category-list'),
            alphaButtons: document.querySelectorAll('.alpha-btn'),
            levelItems: document.querySelectorAll('.level-item'),
            langButtons: document.querySelectorAll('.lang-btn'),
            glossaryContent: document.querySelector('.glossary-content'),
            introDe: document.getElementById('glossary-intro-de'),
            introEn: document.getElementById('glossary-intro-en'),
            statLabels: document.querySelectorAll('.stat-label[data-de]')
        };
    },

    // Initialize the application
    async init() {
        this.cacheDOM();
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
            this.dom.entriesContainer.innerHTML =
                `<div class="loading-state"><p>${this.t('loadError')}</p></div>`;
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
                    // Add to allTags but exclude status markers (not categories)
                    entry.tags.forEach(tag => {
                        if (tag !== 'wip' && tag !== 'contested') this.allTags.add(tag);
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
        this.updateLanguageUI();
        this.renderEntries();
        this.updateStats();
        this.updateAlphabet();
        this.updateCategories();
        this.updateLevels();
    },

    // Update language-dependent UI elements
    updateLanguageUI() {
        // Toggle intro texts
        if (this.dom.introDe && this.dom.introEn) {
            this.dom.introDe.style.display = this.currentLang === 'de' ? 'block' : 'none';
            this.dom.introEn.style.display = this.currentLang === 'en' ? 'block' : 'none';
        }

        // Update stat labels
        this.dom.statLabels.forEach(label => {
            const text = label.dataset[this.currentLang];
            if (text) label.textContent = text;
        });
    },

    // Render filtered entries grouped by letter
    renderEntries() {
        const filtered = this.getFilteredEntries();

        if (filtered.length === 0) {
            this.dom.entriesContainer.innerHTML = `<div class="loading-state"><p>${this.t('noResults')}</p></div>`;
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

        this.dom.entriesContainer.innerHTML = html;

        // Add click handlers for expand/collapse (header and preview text)
        this.dom.entriesContainer.querySelectorAll('.term-header, .term-preview').forEach(clickable => {
            clickable.addEventListener('click', () => {
                clickable.closest('.term-card').classList.toggle('expanded');
            });
        });
    },

    // Render a letter section with entries
    renderLetterSection(letter, entries) {
        const countLabel = entries.length === 1 ? this.t('term') : this.t('terms');
        return `
            <section class="letter-section" id="letter-${letter}">
                <div class="letter-heading">
                    <div class="letter-char">${letter}</div>
                    <div class="letter-meta">${entries.length} ${countLabel}</div>
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
                <h4>${this.t('sources')}</h4>
                <ul>
                    ${entry.sources.map(s => `<li>${this.formatSource(s)}</li>`).join('')}
                </ul>
            </div>
        ` : '';

        // Status badges (contested, wip)
        const isContested = entry.tags.includes('contested');
        const isWip = entry.tags.includes('wip');
        const badgesHtml = (isContested || isWip) ? `
            <div class="term-badges">
                ${isContested ? `<span class="term-badge contested" title="${this.t('contested')}"><i class="fas fa-question-circle"></i> ${this.t('contested')}</span>` : ''}
                ${isWip ? `<span class="term-badge wip"><i class="fas fa-wrench"></i> WIP</span>` : ''}
            </div>
        ` : '';

        // Filter out status tags from displayed categories
        const displayTags = entry.tags.filter(t => t !== 'wip' && t !== 'contested');

        return `
            <article class="term-card" id="${entry.id}">
                <div class="term-header">
                    <div class="term-title-group">
                        <h3 class="term-title">${entry.title}</h3>
                        ${displayTags.length > 0 ? `<div class="term-categories">${displayTags.map(t => `<span class="term-category" data-tag="${t}">${this.formatTagDisplay(t)}</span>`).join('')}</div>` : ''}
                    </div>
                    ${badgesHtml}
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
        this.dom.statTerms.textContent = filtered.length;
        this.dom.statCategories.textContent = this.allTags.size;
    },

    // Update alphabet navigation
    updateAlphabet() {
        const usedLetters = new Set(this.entries.map(e => e.title.charAt(0).toUpperCase()));

        this.dom.alphaButtons.forEach(btn => {
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
        const tagCounts = {};

        this.entries.forEach(entry => {
            entry.tags.forEach(tag => {
                // Exclude status markers from categories
                if (tag !== 'wip' && tag !== 'contested') {
                    tagCounts[tag] = (tagCounts[tag] || 0) + 1;
                }
            });
        });

        const sortedTags = Object.entries(tagCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, this.config.maxCategories);

        const allLabel = this.currentLang === 'de' ? 'Alle' : 'All';
        let html = `
            <div class="category-item active" data-category="all">
                <span>${allLabel}</span>
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

        this.dom.categoryList.innerHTML = html;

        // Re-attach event listeners
        this.dom.categoryList.querySelectorAll('.category-item').forEach(item => {
            item.addEventListener('click', () => {
                this.dom.categoryList.querySelectorAll('.category-item').forEach(i => i.classList.remove('active'));
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

        this.dom.levelItems.forEach(item => {
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
        let searchTimeout;
        this.dom.searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.activeFilters.search = e.target.value;
                this.renderEntries();
                this.updateStats();
            }, this.config.debounceMs);
        });

        // Keyboard shortcut for search
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.dom.searchInput.focus();
            }
        });

        // Level filters
        this.dom.levelItems.forEach(item => {
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
        this.dom.langButtons.forEach(btn => {
            btn.addEventListener('click', async () => {
                const newLang = btn.textContent.trim().toLowerCase();
                if (newLang !== this.currentLang) {
                    this.dom.langButtons.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.currentLang = newLang;
                    this.allTags.clear();
                    await this.loadGlossary();
                }
            });
        });

        // Smooth scroll for alphabet
        this.dom.alphaButtons.forEach(btn => {
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
            this.dom.searchInput.value = '';
            this.dom.categoryList.querySelectorAll('.category-item').forEach(item => {
                item.classList.toggle('active', item.dataset.category === 'all');
            });
            this.dom.levelItems.forEach(item => {
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
            const elementPosition = element.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - this.config.headerOffset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });

            // Highlight effect only for entries
            if (isEntry) {
                element.style.boxShadow = '0 0 0 3px var(--primary-light)';
                setTimeout(() => {
                    element.style.boxShadow = '';
                }, this.config.highlightDurationMs);
            }
        }
    },

    // Handle URL hash on page load
    handleUrlHash() {
        if (window.location.hash) {
            const id = window.location.hash.substring(1);
            setTimeout(() => this.scrollToEntry(id), this.config.hashScrollDelayMs);
        }
    },

    // Filter by specific tag (called when clicking a tag)
    filterByTag(tag) {
        // Update filter
        this.activeFilters.category = tag;

        // Update category list UI
        this.dom.categoryList.querySelectorAll('.category-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.category === tag) {
                item.classList.add('active');
            }
        });

        // Scroll to top of entries
        this.dom.glossaryContent.scrollIntoView({ behavior: 'smooth' });

        // Re-render
        this.renderEntries();
        this.updateStats();
    },

    // Format tag display name (ai-engineering -> AI-Engineering, etc.)
    formatTagDisplay(tag) {
        return this.tagDisplayNames[tag] || this.capitalizeFirst(tag);
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
