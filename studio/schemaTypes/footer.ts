import { defineType, defineField } from "sanity"

export default defineType({
  name: "siteFooter",
  title: "Site Footer",
  type: "document",
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
            // e.g. "/about", "/tell-yours"
          ],
        },
      ],
    }),
  ],
})