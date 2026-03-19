import { toHTML } from "@portabletext/to-html";
import { urlFor } from "./sanityImage";

export function portableTextToHtml(value: any) {
  // Counter resets per article render — gives sequential [1], [2], [3]…
  let fnCount = 0;

  return toHTML(value, {
    components: {
      block: {
        normal: ({ children }: any) => `<p>${children}</p>`,
        h1: ({ children }: any) => `<h1>${children}</h1>`,
        h2: ({ children }: any) => `<h2>${children}</h2>`,
        h3: ({ children }: any) => `<h3>${children}</h3>`,
        h4: ({ children }: any) => `<h4>${children}</h4>`,
        blockquote: ({ children }: any) =>
          `<blockquote>${children}</blockquote>`,
      },
      types: {
        pullQuote: ({ value }: any) => {
          const attribution = value.attribution
            ? `<cite class="pull-quote__attribution">— ${value.attribution}</cite>`
            : "";
          return `
            <aside class="pull-quote">
              <blockquote class="pull-quote__text">${value.quote}</blockquote>
              ${attribution}
            </aside>`;
        },
        imageBlock: ({ value }: any) => {
          if (!value?.image) return "";
          const src = urlFor(value.image).width(1400).quality(85).url();
          const caption = value.caption
            ? `<figcaption class="image-block__caption">${value.caption}</figcaption>`
            : "";
          return `<figure class="image-block image-block--${value.size ?? "normal"}">
    <img src="${src}" alt="${value.caption ?? ""}" loading="lazy" />
    ${caption}
  </figure>`;
        },
      },
      marks: {
        // ─── Footnote ───────────────────────────────────────────────────────
        // Renders as a superscript marker inline. On desktop (≥1280px) the
        // note floats in the right margin, vertically aligned to this spot.
        // On mobile the checkbox trick toggles it open inline below the text.
        footnote: ({ value, children }: any) => {
          fnCount++;
          const n = fnCount;
          const id = `fn-${n}`;
          return `<span class="fn-anchor">${children}<label for="${id}" class="fn-ref" aria-label="Footnote ${n}"><sup class="fn-sup">${n}</sup></label><input type="checkbox" id="${id}" class="fn-toggle" aria-hidden="true" tabindex="-1"><span class="fn-note" role="note"><span class="fn-note-num">${n}</span>${value.note}</span></span>`;
        },

        // ─── Link ────────────────────────────────────────────────────────────
        link: ({ value, children }: any) => {
          const href = value?.href || "#";
          const rel = href.startsWith("http")
            ? "noreferrer noopener"
            : undefined;
          const target = href.startsWith("http") ? "_blank" : undefined;
          return `<a href="${href}" rel="${rel ?? ""}" target="${target ?? ""}">${children}</a>`;
        },
      },
    },
  });
}
