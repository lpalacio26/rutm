import { defineType, defineField } from "sanity"

export default defineType({
  name: "newsletterPage",
  title: "Newsletter Page",
  type: "document",
  preview: {
    prepare() {
      return { title: "Newsletter" };
    },
  },
  fields: [
    defineField({
      name: "body",
      title: "Body text",
      description: "Introductory text shown above the subscription form.",
      type: "array",
      of: [{ type: "block" }],
    }),
    defineField({
      name: "marqueeText",
      title: "Marquee banner text",
      type: "string",
      initialValue: "NEWSLETTER",
    }),
    defineField({
      name: "marqueeSpeed",
      title: "Marquee speed (px/s)",
      type: "number",
      initialValue: 40,
    }),
  ],
})
