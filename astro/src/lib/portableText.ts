import { toHTML } from "@portabletext/to-html";

export function portableTextToHtml(value: any) {
  return toHTML(value, {
    components: {
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
