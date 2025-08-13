// Mobile menu toggle
const mobileToggle = document.querySelector('.mobile-menu-toggle');
const mainNav = document.querySelector('.main-nav');

if (mobileToggle) {
   mobileToggle.addEventListener('click', function() {
       const isExpanded = this.getAttribute('aria-expanded') === 'true';
       this.setAttribute('aria-expanded', !isExpanded);
       mainNav.classList.toggle('active');
   });
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
   anchor.addEventListener('click', function (e) {
       e.preventDefault();
       const targetId = this.getAttribute('href');
       const target = document.querySelector(targetId);
       
       if (target) {
           target.scrollIntoView({ 
               behavior: 'smooth', 
               block: 'start' 
           });
           
           // Update URL without jumping
           history.pushState(null, null, targetId);
           
           // Focus management for accessibility
           const focusableElement = target.querySelector('h2, h3, [tabindex]') || target;
           focusableElement.focus({ preventScroll: true });
           
           // Close mobile menu if open
           if (mainNav && mainNav.classList.contains('active')) {
               mainNav.classList.remove('active');
               mobileToggle.setAttribute('aria-expanded', 'false');
           }
       }
   });
});

// Horizontal scroll for session cards
document.querySelectorAll('.session-cards-container').forEach(container => {
   const scrollContainer = container.querySelector('.session-cards');
   const leftBtn = container.querySelector('.scroll-btn-left');
   const rightBtn = container.querySelector('.scroll-btn-right');
   
   if (scrollContainer && leftBtn && rightBtn) {
       // Scroll amount (width of one card plus gap)
       const scrollAmount = 300;
       
       // Update button states based on scroll position
       function updateButtonStates() {
           const scrollLeft = scrollContainer.scrollLeft;
           const maxScroll = scrollContainer.scrollWidth - scrollContainer.clientWidth;
           
           // Disable/enable left button
           if (scrollLeft <= 0) {
               leftBtn.disabled = true;
               leftBtn.style.opacity = '0.3';
           } else {
               leftBtn.disabled = false;
               leftBtn.style.opacity = '1';
           }
           
           // Disable/enable right button
           if (scrollLeft >= maxScroll - 1) {
               rightBtn.disabled = true;
               rightBtn.style.opacity = '0.3';
           } else {
               rightBtn.disabled = false;
               rightBtn.style.opacity = '1';
           }
       }
       
       // Initial button state
       updateButtonStates();
       
       // Scroll left
       leftBtn.addEventListener('click', () => {
           scrollContainer.scrollBy({
               left: -scrollAmount,
               behavior: 'smooth'
           });
       });
       
       // Scroll right
       rightBtn.addEventListener('click', () => {
           scrollContainer.scrollBy({
               left: scrollAmount,
               behavior: 'smooth'
           });
       });
       
       // Update button states on scroll
       scrollContainer.addEventListener('scroll', updateButtonStates);
       
       // Keyboard navigation for cards
       scrollContainer.addEventListener('keydown', (e) => {
           if (e.key === 'ArrowLeft') {
               scrollContainer.scrollBy({
                   left: -scrollAmount,
                   behavior: 'smooth'
               });
           } else if (e.key === 'ArrowRight') {
               scrollContainer.scrollBy({
                   left: scrollAmount,
                   behavior: 'smooth'
               });
           }
       });
   }
});

// Active section highlighting in TOC
function highlightActiveSection() {
   const sections = document.querySelectorAll('.course-section');
   const tocLinks = document.querySelectorAll('.toc a');
   
   // Get current scroll position
   const scrollPosition = window.scrollY + 100;
   
   sections.forEach((section, index) => {
       const sectionTop = section.offsetTop;
       const sectionHeight = section.offsetHeight;
       
       if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
           // Remove active class from all TOC links
           tocLinks.forEach(link => link.classList.remove('active'));
           
           // Add active class to current section link
           const currentLink = document.querySelector(`.toc a[href="#${section.id}"]`);
           if (currentLink) {
               currentLink.classList.add('active');
           }
       }
   });
}

// Throttle function for scroll performance
function throttle(func, limit) {
   let inThrottle;
   return function() {
       const args = arguments;
       const context = this;
       if (!inThrottle) {
           func.apply(context, args);
           inThrottle = true;
           setTimeout(() => inThrottle = false, limit);
       }
   }
}

// Apply throttled scroll listener
window.addEventListener('scroll', throttle(highlightActiveSection, 100));

// Lazy loading for images
if ('IntersectionObserver' in window) {
   const imageObserver = new IntersectionObserver((entries, observer) => {
       entries.forEach(entry => {
           if (entry.isIntersecting) {
               const img = entry.target;
               // Only load if src is different from a placeholder
               if (img.dataset.src && img.src !== img.dataset.src) {
                   img.src = img.dataset.src;
                   img.classList.add('loaded');
                   observer.unobserve(img);
               }
           }
       });
   }, {
       rootMargin: '50px 0px',
       threshold: 0.01
   });
   
   // Observe all card images
   document.querySelectorAll('.card-image img').forEach(img => {
       if (img.dataset.src) {
           imageObserver.observe(img);
       }
   });
}

// Print functionality
function setupPrintHandling() {
   // Before print - expand all content
   window.addEventListener('beforeprint', () => {
       document.body.classList.add('printing');
       // Ensure all cards are visible
       document.querySelectorAll('.session-cards').forEach(container => {
           container.style.overflow = 'visible';
       });
   });
   
   // After print - restore normal view
   window.addEventListener('afterprint', () => {
       document.body.classList.remove('printing');
       document.querySelectorAll('.session-cards').forEach(container => {
           container.style.overflow = '';
       });
   });
}

setupPrintHandling();

// Keyboard navigation improvements
document.addEventListener('keydown', (e) => {
   // Escape key closes mobile menu
   if (e.key === 'Escape' && mainNav && mainNav.classList.contains('active')) {
       mainNav.classList.remove('active');
       mobileToggle.setAttribute('aria-expanded', 'false');
       mobileToggle.focus();
   }
});

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
   // Set initial active section in TOC
   highlightActiveSection();
   
   // Add loaded class to body for CSS animations
   document.body.classList.add('loaded');
   
   // Check for direct link to section
   if (window.location.hash) {
       setTimeout(() => {
           const target = document.querySelector(window.location.hash);
           if (target) {
               target.scrollIntoView({ behavior: 'smooth', block: 'start' });
           }
       }, 100);
   }
});

// Handle browser back/forward navigation
window.addEventListener('popstate', () => {
   if (window.location.hash) {
       const target = document.querySelector(window.location.hash);
       if (target) {
           target.scrollIntoView({ behavior: 'smooth', block: 'start' });
       }
   }
});

// Touch swipe support for mobile card scrolling
let touchStartX = 0;
let touchEndX = 0;

document.querySelectorAll('.session-cards').forEach(container => {
   container.addEventListener('touchstart', (e) => {
       touchStartX = e.changedTouches[0].screenX;
   }, { passive: true });
   
   container.addEventListener('touchend', (e) => {
       touchEndX = e.changedTouches[0].screenX;
       handleSwipe(container);
   }, { passive: true });
});

function handleSwipe(container) {
   const swipeThreshold = 50;
   const diff = touchStartX - touchEndX;
   
   if (Math.abs(diff) > swipeThreshold) {
       if (diff > 0) {
           // Swiped left - scroll right
           container.scrollBy({ left: 300, behavior: 'smooth' });
       } else {
           // Swiped right - scroll left
           container.scrollBy({ left: -300, behavior: 'smooth' });
       }
   }
}

// Performance: Reduce reflows by batching DOM reads/writes
function optimizeScrollPerformance() {
   let ticking = false;
   
   function updateScrollProgress() {
       const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
       const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
       const scrolled = (winScroll / height) * 100;
       
       // Update any scroll progress indicators here if needed
       
       ticking = false;
   }
   
   function requestTick() {
       if (!ticking) {
           window.requestAnimationFrame(updateScrollProgress);
           ticking = true;
       }
   }
   
   window.addEventListener('scroll', requestTick, { passive: true });
}

optimizeScrollPerformance();