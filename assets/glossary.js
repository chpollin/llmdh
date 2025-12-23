/**
 * Glossary Application
 * Custom parser for structured Markdown glossary format
 */

// Global state
let allEntries = [];
let currentLang = 'en';
let activeFilters = {
    search: '',
    level: 'all',
    tags: new Set(),
    hideWip: false
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

/**
 * Initialize the glossary application
 */
async function initializeApp() {
    // Load initial language
    const savedLang = localStorage.getItem('glossaryLang') || 'en';
    currentLang = savedLang;
    updateLanguageButtons();

    // Setup event listeners
    setupEventListeners();

    // Load glossary data
    await loadGlossary(currentLang);

    // Handle URL fragment (e.g., #llm)
    handleUrlFragment();
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Language switcher
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const lang = e.target.dataset.lang;
            switchLanguage(lang);
        });
    });

    // Search input
    const searchInput = document.getElementById('search-input');
    searchInput.addEventListener('input', debounce((e) => {
        activeFilters.search = e.target.value.toLowerCase();
        applyFilters();
    }, 300));

    // Level filters
    document.querySelectorAll('input[name="level"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            activeFilters.level = e.target.value;
            applyFilters();
        });
    });

    // Hide WIP checkbox
    const hideWipCheckbox = document.getElementById('hide-wip');
    hideWipCheckbox.addEventListener('change', (e) => {
        activeFilters.hideWip = e.target.checked;
        applyFilters();
    });

    // Reset filters button
    document.getElementById('reset-filters').addEventListener('click', resetFilters);
}

/**
 * Load glossary from Markdown file
 */
async function loadGlossary(lang) {
    const filename = lang === 'de' ? 'glossar_de.md' : 'glossary_en.md';
    const url = `${filename}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load ${filename}`);
        }

        const markdown = await response.text();
        allEntries = parseGlossaryMarkdown(markdown);

        // Build tag filters
        buildTagFilters();

        // Render all entries
        applyFilters();

    } catch (error) {
        console.error('Error loading glossary:', error);
        showError(`Could not load glossary file: ${filename}`);
    }
}

/**
 * Parse Markdown into structured glossary entries
 */
function parseGlossaryMarkdown(markdown) {
    const entries = [];

    // Split by ## headers (new entries)
    const blocks = markdown.split(/\n(?=## )/);

    blocks.forEach(block => {
        if (!block.trim() || !block.startsWith('##')) return;

        const lines = block.split('\n');
        const entry = {};

        // Line 0: Title (remove ##)
        entry.title = lines[0].replace(/^##\s*/, '').trim();

        // Parse metadata lines
        let bodyStartIndex = 1;
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();

            if (line.startsWith('id:')) {
                entry.id = line.replace('id:', '').trim();
            } else if (line.startsWith('tags:')) {
                const tagString = line.replace('tags:', '').trim();
                entry.tags = tagString.split(',').map(t => t.trim()).filter(t => t);
            } else if (line.startsWith('level:')) {
                entry.level = line.replace('level:', '').trim();
            } else if (line === '') {
                // Empty line marks end of metadata
                bodyStartIndex = i + 1;
                break;
            }
        }

        // Validate required fields
        if (!entry.id || !entry.title || !entry.tags || !entry.level) {
            console.warn('Skipping invalid entry:', entry.title);
            return;
        }

        // Extract body text and sources
        const remainingLines = lines.slice(bodyStartIndex);
        const bodyLines = [];
        const sourceLines = [];
        let inSources = false;

        remainingLines.forEach(line => {
            if (line.trim().startsWith('* ')) {
                inSources = true;
                sourceLines.push(line.trim().substring(2)); // Remove "* "
            } else if (!inSources && line.trim()) {
                bodyLines.push(line.trim());
            } else if (inSources && line.trim() && !line.trim().startsWith('* ')) {
                // Continuation of a source line
                sourceLines[sourceLines.length - 1] += ' ' + line.trim();
            }
        });

        entry.body = bodyLines.join(' ').trim();
        entry.sources = sourceLines;

        // Check for WIP tag
        entry.isWip = entry.tags.includes('wip');

        entries.push(entry);
    });

    // Sort alphabetically by title
    entries.sort((a, b) => a.title.localeCompare(b.title));

    return entries;
}

/**
 * Build tag filter checkboxes from all available tags
 */
function buildTagFilters() {
    const tagContainer = document.getElementById('tag-filters');
    tagContainer.innerHTML = '';

    // Collect all unique tags (excluding 'wip')
    const tagCounts = {};
    allEntries.forEach(entry => {
        entry.tags.forEach(tag => {
            if (tag !== 'wip') {
                tagCounts[tag] = (tagCounts[tag] || 0) + 1;
            }
        });
    });

    // Sort tags alphabetically
    const sortedTags = Object.keys(tagCounts).sort();

    // Create checkbox for each tag
    sortedTags.forEach(tag => {
        const div = document.createElement('div');
        div.className = 'tag-filter';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `tag-${tag}`;
        checkbox.value = tag;
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                activeFilters.tags.add(tag);
            } else {
                activeFilters.tags.delete(tag);
            }
            applyFilters();
        });

        const label = document.createElement('label');
        label.htmlFor = `tag-${tag}`;
        label.textContent = formatTagDisplay(tag);

        const count = document.createElement('span');
        count.className = 'tag-count';
        count.textContent = `(${tagCounts[tag]})`;

        div.appendChild(checkbox);
        div.appendChild(label);
        div.appendChild(count);
        tagContainer.appendChild(div);
    });
}

/**
 * Apply all active filters and render results
 */
function applyFilters() {
    let filtered = [...allEntries];

    // Filter by search term
    if (activeFilters.search) {
        filtered = filtered.filter(entry => {
            const searchTerm = activeFilters.search;
            return entry.title.toLowerCase().includes(searchTerm) ||
                   entry.body.toLowerCase().includes(searchTerm) ||
                   entry.tags.some(tag => tag.toLowerCase().includes(searchTerm));
        });
    }

    // Filter by level
    if (activeFilters.level !== 'all') {
        if (activeFilters.level === 'basic') {
            filtered = filtered.filter(e => e.level === 'basic');
        } else if (activeFilters.level === 'basic-intermediate') {
            filtered = filtered.filter(e => e.level === 'basic' || e.level === 'intermediate');
        }
    }

    // Filter by tags (OR logic: show if ANY selected tag matches)
    if (activeFilters.tags.size > 0) {
        filtered = filtered.filter(entry => {
            return entry.tags.some(tag => activeFilters.tags.has(tag));
        });
    }

    // Filter WIP
    if (activeFilters.hideWip) {
        filtered = filtered.filter(e => !e.isWip);
    }

    // Render results
    renderEntries(filtered);
    updateResultsCount(filtered.length, allEntries.length);
}

/**
 * Render glossary entries
 */
function renderEntries(entries) {
    const container = document.getElementById('glossary-entries');
    const noResults = document.getElementById('no-results');

    if (entries.length === 0) {
        container.innerHTML = '';
        noResults.style.display = 'block';
        return;
    }

    noResults.style.display = 'none';
    container.innerHTML = '';

    entries.forEach(entry => {
        const card = createEntryCard(entry);
        container.appendChild(card);
    });
}

/**
 * Create HTML card for a glossary entry
 */
function createEntryCard(entry) {
    const card = document.createElement('article');
    card.className = 'entry-card';
    card.id = entry.id;

    // Header
    const header = document.createElement('div');
    header.className = 'entry-header';

    const title = document.createElement('h2');
    title.className = 'entry-title';
    title.textContent = entry.title;

    const meta = document.createElement('div');
    meta.className = 'entry-meta';

    const levelBadge = document.createElement('span');
    levelBadge.className = `level-badge ${entry.level}`;
    levelBadge.textContent = entry.level;
    meta.appendChild(levelBadge);

    if (entry.isWip) {
        const wipBadge = document.createElement('span');
        wipBadge.className = 'wip-badge';
        wipBadge.textContent = 'WIP';
        meta.appendChild(wipBadge);
    }

    header.appendChild(title);
    header.appendChild(meta);
    card.appendChild(header);

    // Tags
    const tagsDiv = document.createElement('div');
    tagsDiv.className = 'entry-tags';
    entry.tags.forEach(tag => {
        const tagSpan = document.createElement('span');
        tagSpan.className = tag === 'wip' ? 'tag tag-wip' : 'tag';
        tagSpan.textContent = formatTagDisplay(tag);
        tagSpan.dataset.tag = tag;
        tagSpan.style.cursor = 'pointer';
        tagSpan.addEventListener('click', () => filterByTag(tag));
        tagsDiv.appendChild(tagSpan);
    });
    card.appendChild(tagsDiv);

    // Body
    const body = document.createElement('div');
    body.className = 'entry-body';
    body.innerHTML = linkifyTerms(entry.body, entry.id);
    card.appendChild(body);

    // Sources
    if (entry.sources && entry.sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'entry-sources';

        const sourcesTitle = document.createElement('h4');
        sourcesTitle.textContent = currentLang === 'de' ? 'Quellen' : 'Sources';
        sourcesDiv.appendChild(sourcesTitle);

        const sourcesList = document.createElement('ul');
        entry.sources.forEach(source => {
            const li = document.createElement('li');
            li.innerHTML = source; // Allows markdown links in sources
            sourcesList.appendChild(li);
        });
        sourcesDiv.appendChild(sourcesList);
        card.appendChild(sourcesDiv);
    }

    return card;
}

/**
 * Auto-link terms mentioned in body text
 */
function linkifyTerms(text, currentEntryId) {
    let linkedText = text;

    // Create a map of all entry titles to their IDs
    const termMap = {};
    allEntries.forEach(entry => {
        if (entry.id !== currentEntryId) {
            termMap[entry.title] = entry.id;
        }
    });

    // Sort by length (longest first) to avoid partial matches
    const sortedTerms = Object.keys(termMap).sort((a, b) => b.length - a.length);

    // Replace first occurrence of each term with a link
    sortedTerms.forEach(term => {
        const regex = new RegExp(`\\b(${escapeRegex(term)})\\b`, 'i');
        const match = linkedText.match(regex);
        if (match) {
            const id = termMap[term];
            linkedText = linkedText.replace(
                regex,
                `<a href="#${id}" onclick="event.preventDefault(); scrollToEntry('${id}')">$1</a>`
            );
        }
    });

    return linkedText;
}

/**
 * Scroll to and highlight an entry
 */
function scrollToEntry(entryId) {
    const element = document.getElementById(entryId);
    if (element) {
        // Update URL
        history.pushState(null, null, `#${entryId}`);

        // Scroll to element
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Add highlight effect
        element.classList.add('highlighted');
        setTimeout(() => {
            element.classList.remove('highlighted');
        }, 2000);
    }
}

/**
 * Handle URL fragment on page load
 */
function handleUrlFragment() {
    if (window.location.hash) {
        const entryId = window.location.hash.substring(1);
        setTimeout(() => {
            scrollToEntry(entryId);
        }, 500); // Wait for rendering
    }
}

/**
 * Switch language
 */
async function switchLanguage(lang) {
    if (lang === currentLang) return;

    currentLang = lang;
    localStorage.setItem('glossaryLang', lang);
    updateLanguageButtons();

    // Reload glossary
    await loadGlossary(lang);
}

/**
 * Update language button states
 */
function updateLanguageButtons() {
    document.querySelectorAll('.lang-btn').forEach(btn => {
        if (btn.dataset.lang === currentLang) {
            btn.classList.add('active');
            btn.setAttribute('aria-pressed', 'true');
        } else {
            btn.classList.remove('active');
            btn.setAttribute('aria-pressed', 'false');
        }
    });
}

/**
 * Reset all filters
 */
function resetFilters() {
    // Clear search
    document.getElementById('search-input').value = '';
    activeFilters.search = '';

    // Reset level
    document.querySelector('input[name="level"][value="all"]').checked = true;
    activeFilters.level = 'all';

    // Clear tags
    document.querySelectorAll('#tag-filters input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    activeFilters.tags.clear();

    // Clear WIP
    document.getElementById('hide-wip').checked = false;
    activeFilters.hideWip = false;

    applyFilters();
}

/**
 * Update results count display
 */
function updateResultsCount(shown, total) {
    const countElement = document.getElementById('results-count');
    if (shown === total) {
        countElement.textContent = `Showing all ${total} entries`;
    } else {
        countElement.textContent = `Showing ${shown} of ${total} entries`;
    }
}

/**
 * Show error message
 */
function showError(message) {
    const container = document.getElementById('glossary-entries');
    container.innerHTML = `
        <div style="padding: 40px; text-align: center; color: #999;">
            <p style="font-size: 1.1em; margin-bottom: 10px;">⚠️ Error</p>
            <p>${message}</p>
        </div>
    `;
}

/**
 * Debounce helper for search input
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Escape special regex characters
 */
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Format tag display name (ai-engineering -> AI-Engineering, etc.)
 */
function formatTagDisplay(tag) {
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
    return displayNames[tag] || tag.charAt(0).toUpperCase() + tag.slice(1);
}

/**
 * Filter by specific tag (called when clicking a tag)
 */
function filterByTag(tag) {
    // Clear existing tag filters and set only this one
    activeFilters.tags.clear();
    activeFilters.tags.add(tag);

    // Update checkboxes
    document.querySelectorAll('#tag-filters input[type="checkbox"]').forEach(cb => {
        cb.checked = cb.value === tag;
    });

    // Scroll to top
    document.getElementById('glossary-entries').scrollIntoView({ behavior: 'smooth' });

    applyFilters();
}
