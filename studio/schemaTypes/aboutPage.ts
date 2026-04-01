import { defineType, defineField } from "sanity"

export default defineType({
  name: "aboutPage",
  title: "About Page",
  type: "document",
  preview: {
    prepare() {
      return { title: "About" };
    },
  },
  fields: [
    defineField({
      name: "definitionBlocks",
      title: "Definition blockquotes",
      description: "The quoted definitions — etymology first, then Cambridge entries",
      type: "array",
      of: [{ type: "text" }],
    }),
    defineField({
      name: "dictionarySource",
      title: "Dictionary source attribution",
      type: "string",
      initialValue: "Cambridge Advanced Learner's Dictionary & Thesaurus © Cambridge University Press",
    }),
    defineField({
      name: "bodyEN",
      title: "Body text (English)",
      type: "array",
      of: [
        {
          type: "block",
          marks: {
            annotations: [
              {
                name: "link",
                type: "object",
                title: "Link",
                fields: [{ name: "href", type: "url", title: "URL" }],
              },
              {
                name: "footnote",
                type: "object",
                title: "Footnote",
                fields: [
                  {
                    name: "note",
                    title: "Note",
                    type: "text",
                    rows: 3,
                  },
                ],
              },
            ],
          },
        },
      ],
    }),
    defineField({
      name: "bodyFR",
      title: "Body text (French)",
      type: "array",
      of: [
        {
          type: "block",
          marks: {
            annotations: [
              {
                name: "link",
                type: "object",
                title: "Link",
                fields: [{ name: "href", type: "url", title: "URL" }],
              },
              {
                name: "footnote",
                type: "object",
                title: "Footnote",
                fields: [
                  {
                    name: "note",
                    title: "Note",
                    type: "text",
                    rows: 3,
                  },
                ],
              },
            ],
          },
        },
      ],
    }),
  ],
})