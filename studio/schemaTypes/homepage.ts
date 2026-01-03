export default {
  name: 'homepage',
  title: 'Homepage',
  type: 'document',
  fields: [
    {
      name: 'marqueeText',
      title: 'Marquee text',
      type: 'string',
      validation: (Rule: any) => Rule.required(),
    },
    {
      name: 'marqueeSpeed',
      title: 'Marquee speed (px / second)',
      type: 'number',
      initialValue: 40,
      validation: (Rule: any) => Rule.min(5).max(200),
    },
  ],
}
