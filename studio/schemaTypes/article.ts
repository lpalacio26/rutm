import { defineType, defineField } from "sanity";

export default defineType({
  name: "article",
  title: "Article",
  type: "document",
  fields: [
    defineField({
      name: "title",
      title: "Title",
      type: "string",
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: "slug",
      title: "Slug",
      type: "slug",
      options: { source: "title", maxLength: 96 },
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: "excerpt",
      title: "Excerpt",
      type: "text",
    }),
    defineField({
      name: "format",
      title: "Article type",
      type: "string",
      options: {
        list: [
          { title: "Feature", value: "feature" },
          { title: "Interview", value: "interview" },
          { title: "Review", value: "review" },
          { title: "News", value: "news" },
          { title: "Essay", value: "essay" },
        ],
        layout: "radio",
      },
      initialValue: "feature",
    }),
    defineField({
      name: "heroImage",
      title: "Hero image",
      type: "image",
      options: { hotspot: true },
    }),
    defineField({
      name: "content",
      title: "Content",
      type: "array",
      of: [{ type: "block" }],
    }),

    // Relations
    defineField({
      name: "author",
      title: "Author",
      type: "reference",
      to: [{ type: "author" }],
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: "section",
      title: "Section",
      type: "reference",
      to: [{ type: "section" }],
      validation: (Rule) => Rule.required(),
    }),
    defineField({
      name: "themes",
      title: "Themes",
      type: "array",
      of: [{ type: "reference", to: [{ type: "theme" }] }],
    }),

    // Publishing
    defineField({
      name: "publishedAt",
      title: "Published at",
      type: "datetime",
    }),
    defineField({
      name: "featured",
      title: "Featured on homepage",
      type: "boolean",
      initialValue: false,
    }),
  ],
});
