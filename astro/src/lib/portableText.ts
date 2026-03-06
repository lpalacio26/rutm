import { toHTML } from "@portabletext/to-html";

export function portableTextToHtml(value: any) {
  return toHTML(value, {
    components: {
      block: {
        normal: ({ children }: any) => `<p>${children}</p>`,
        h1: ({ children }: any) => `<h1>${children}</h1>`,
        h2: ({ children }: any) => `<h2>${children}</h2>`,
        h3: ({ children }: any) => `<h3>${children}</h3>`,
        h4: ({ children }: any) => `<h4>${children}</h4>`,
        blockquote: ({ children }: any) => `<blockquote>${children}</blockquote>`,
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
      },
      marks: {
        link: ({ value, children }: any) => {
          const href = value?.href || "#";
          const rel = href.startsWith("http") ? "noreferrer noopener" : undefined;
          const target = href.startsWith("http") ? "_blank" : undefined;
          return `<a href="${href}" rel="${rel ?? ""}" target="${target ?? ""}">${children}</a>`;
        },
      },
    },
  });
}