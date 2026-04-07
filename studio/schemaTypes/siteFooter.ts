import { defineType, defineField } from "sanity"

export default defineType({
  name: "siteFooter",
  title: "Site Footer",
  type: "document",
  preview: {
    prepare() {
      return { title: "Footer" };
    },
  },
  fields: [
    defineField({
      name: "links",
      title: "Footer links",
      type: "array",
      of: [
        {
          type: "object",
          fields: [
            { name: "label", title: "Label", type: "string" },
            { name: "url", title: "URL path", type: "string" },
          ],
        },
      ],
    }),
    defineField({
      name: "socialLinks",
      title: "Social Media",
      description: "Social media accounts — displayed with an arrow (↗) automatically.",
      type: "array",
      of: [
        {
          type: "object",
          preview: {
            select: { title: "name", subtitle: "url" },
          },
          fields: [
            {
              name: "order",
              title: "Order",
              type: "number",
              description: "Controls display order (lowest first).",
              validation: (Rule: any) => Rule.required().integer().min(1),
            },
            {
              name: "name",
              title: "Name",
              type: "string",
              description: 'e.g. "Instagram", "X", "TikTok"',
              validation: (Rule: any) => Rule.required(),
            },
            {
              name: "url",
              title: "URL",
              type: "url",
              description: "Full URL including https://",
              validation: (Rule: any) =>
                Rule.required().uri({ scheme: ["http", "https"] }),
            },
          ],
        },
      ],
    }),
  ],
})