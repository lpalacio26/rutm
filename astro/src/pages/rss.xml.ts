import rss from "@astrojs/rss";
import { sanity } from "../lib/sanity";

export async function GET(context: { site: URL }) {
  const articles = await sanity.fetch<
    { title: string; excerpt?: string; publishedAt?: string; slug: string }[]
  >(
    `*[_type=="article"] | order(publishedAt desc){
      title,
      excerpt,
      publishedAt,
      "slug": slug.current
    }`
  );

  return rss({
    title: "RÜTM",
    description: "RÜTM Magazine",
    site: context.site,
    items: articles.map((a) => ({
      title: a.title,
      link: `/article/${a.slug}`,
      description: a.excerpt,
      pubDate: a.publishedAt ? new Date(a.publishedAt) : undefined,
    })),
  });
}
